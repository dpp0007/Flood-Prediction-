"""
Autonomous Flood Early Warning Prediction System - Main Entry Point.

Enables continuous operation with fixed-interval pipeline execution.

Data Source: Central Water Commission (CWC), Government of India
Website: https://ffs.india-water.gov.in/

Usage:
    python autonomous_main.py                    # Run in autonomous mode
    python autonomous_main.py --interval 60      # Custom interval (minutes)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import sys
import argparse
from typing import Dict

from src.utils.logger import setup_logger
from src.utils.constants import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_NAMES, WEATHER_ENABLED
from src.utils.output_manager import OutputManager
from src.utils.scheduler import AutonomousScheduler, CycleStatistics
from src.utils.alert_manager import AlertManager
from src.data_engineering import DataCleaner
from src.features import FeatureGenerator
from src.models import ModelTrainer
from src.inference import FloodPredictor, AlertGenerator
from src.explainability import ExplanationGenerator

# Weather-aware imports (NEW)
from src.weather.gfs_fetcher import GFSFetcher
from src.weather.weather_cache import WeatherCache
from src.features.weather_aware_feature_generator import WeatherAwareFeatureGenerator
from src.inference.weather_aware_predictor import WeatherAwarePredictor
from src.utils.weather_output_manager import WeatherAwareOutputManager

logger = setup_logger(__name__)


class AutonomousFloodPredictionPipeline:
    """
    Autonomous flood prediction pipeline for continuous operation.
    """
    
    def __init__(self, use_weather_aware: bool = True):
        """
        Initialize pipeline.
        
        Args:
            use_weather_aware: If True, use weather-aware mode (GFS + hydro)
                              If False, use hydro-only mode (backward compatible)
        """
        self.use_weather_aware = use_weather_aware
        self.cleaner = DataCleaner()
        
        # Choose feature generator based on mode
        if use_weather_aware and WEATHER_ENABLED:
            self.feature_gen = WeatherAwareFeatureGenerator()
            self.predictor_class = WeatherAwarePredictor
            self.mode = "WEATHER-AWARE"
            logger.info("✅ WEATHER-AWARE MODE ENABLED (GFS + Hydro)")
        else:
            self.feature_gen = FeatureGenerator()
            self.predictor_class = FloodPredictor
            self.mode = "HYDRO-ONLY"
            logger.info("✅ HYDRO-ONLY MODE (Backward Compatible)")
        
        self.trainer = ModelTrainer()
        self.predictor = None  # Will be initialized later
        self.alert_gen = AlertGenerator()
        self.explainer = ExplanationGenerator()
        self.output_manager = OutputManager()
        self.alert_manager = AlertManager()
        
        # Weather components (if weather-aware)
        self.gfs_fetcher = None
        self.weather_cache = None
        if use_weather_aware and WEATHER_ENABLED:
            self.gfs_fetcher = GFSFetcher(cache_dir=Path("data/cache"))
            self.weather_cache = WeatherCache(cache_dir=Path("data/cache"))
        
        logger.info(f"AutonomousFloodPredictionPipeline initialized in {self.mode} mode")
    
    def load_raw_data(self, csv_path: str) -> pd.DataFrame:
        """Load raw CWC data from CSV."""
        df = pd.read_csv(csv_path)
        return df
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        cleaned = self.cleaner.clean(raw_data)
        return cleaned
    
    def generate_features(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML features (hydro-only or weather-aware)."""
        data_dict = {
            'stations': cleaned_data[['station_id', 'station_name', 'river_name', 'basin', 'state']].drop_duplicates(),
            'levels': cleaned_data
        }
        
        # If weather-aware mode, try to fetch weather data
        weather_data = None
        if self.mode == "WEATHER-AWARE":
            try:
                logger.info("Fetching GFS rainfall forecasts...")
                stations = data_dict['stations'][['station_id', 'latitude', 'longitude']].drop_duplicates()
                weather_data = self.gfs_fetcher.fetch_rainfall_forecast(stations, forecast_hours=24)
                
                if weather_data is not None and not weather_data.empty:
                    logger.info(f"✅ GFS data fetched for {len(weather_data)} stations")
                else:
                    logger.warning("⚠️ GFS fetch returned empty data, falling back to hydro-only")
                    weather_data = None
            except Exception as e:
                logger.warning(f"⚠️ GFS fetch failed: {str(e)}, falling back to hydro-only")
                weather_data = None
            
            # Generate weather-aware features
            features = self.feature_gen.generate(data_dict, weather_data=weather_data)
        else:
            # Generate hydro-only features
            features = self.feature_gen.generate(data_dict)
        
        return features
    
    def create_labels(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic labels for training."""
        labels = pd.DataFrame()
        
        distance_to_danger = features.get('distance_to_danger', 0)
        labels['risk_score'] = np.clip(1.0 - (distance_to_danger / 5.0), 0, 1)
        
        threshold = distance_to_danger.quantile(0.2)
        labels['warning_label'] = (distance_to_danger < threshold).astype(int)
        
        labels['risk_tier'] = pd.cut(labels['risk_score'], 
                                     bins=[0, 0.33, 0.67, 1.0],
                                     labels=[0, 1, 2]).astype(int)
        
        return labels
    
    def train_models(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict:
        """Train prediction models."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train_risk, y_test_risk = train_test_split(
            features, labels['risk_score'], test_size=0.2, random_state=42
        )
        _, _, y_train_warn, y_test_warn = train_test_split(
            features, labels['warning_label'], test_size=0.2, random_state=42
        )
        _, _, y_train_tier, y_test_tier = train_test_split(
            features, labels['risk_tier'], test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[[col for col in FEATURE_NAMES if col in X_train.columns]])
        X_test_scaled = scaler.transform(X_test[[col for col in FEATURE_NAMES if col in X_test.columns]])
        
        X_train = pd.DataFrame(X_train_scaled, columns=[col for col in FEATURE_NAMES if col in X_train.columns])
        X_test = pd.DataFrame(X_test_scaled, columns=[col for col in FEATURE_NAMES if col in X_test.columns])
        
        results = {}
        
        risk_results = self.trainer.train_risk_regressor(
            X_train, y_train_risk,
            X_test, y_test_risk
        )
        results['risk_regressor'] = risk_results['metrics']
        
        warning_results = self.trainer.train_warning_classifier(
            X_train, y_train_warn,
            X_test, y_test_warn
        )
        results['warning_classifier'] = warning_results['metrics']
        
        tier_results = self.trainer.train_risk_tier_classifier(
            X_train, y_train_tier,
            X_test, y_test_tier
        )
        results['risk_tier_classifier'] = tier_results['metrics']
        
        self.trainer.save_models(str(MODELS_DIR))
        
        return results
    
    def generate_predictions(self, features: pd.DataFrame, weather_data: pd.DataFrame = None) -> pd.DataFrame:
        """Generate predictions using trained models."""
        if self.mode == "WEATHER-AWARE":
            self.predictor = WeatherAwarePredictor(model_dir=str(MODELS_DIR))
            predictions = self.predictor.predict_with_weather(features, weather_data=weather_data)
            logger.info(f"✅ Generated {len(predictions)} predictions in WEATHER-AWARE mode")
        else:
            self.predictor = FloodPredictor(model_dir=str(MODELS_DIR))
            predictions = self.predictor.predict_all(features)
            logger.info(f"✅ Generated {len(predictions)} predictions in HYDRO-ONLY mode")
        
        return predictions
    
    def run_cycle(self, cycle_number: int = 1) -> Dict:
        """
        Run a single prediction cycle.
        
        Args:
            cycle_number: Cycle number for logging
            
        Returns:
            Cycle statistics dictionary
        """
        cycle_stats = CycleStatistics(cycle_number)
        
        try:
            # Print mode at start of cycle
            logger.info("="*70)
            logger.info(f"CYCLE {cycle_number} - MODE: {self.mode}")
            logger.info("="*70)
            
            # Find latest data file
            raw_files = list(RAW_DATA_DIR.glob('cwc_hydrograph_*.csv'))
            if not raw_files:
                logger.error("No CWC hydrograph data found")
                return {}
            
            data_file = str(sorted(raw_files)[-1])
            
            # Load and process data
            raw_data = self.load_raw_data(data_file)
            cleaned_data = self.clean_data(raw_data)
            features = self.generate_features(cleaned_data)
            labels = self.create_labels(features)
            
            # Train models
            self.train_models(features, labels)
            
            # Generate predictions (with weather data if available)
            weather_data = None
            if self.mode == "WEATHER-AWARE" and self.gfs_fetcher:
                try:
                    stations = cleaned_data[['station_id', 'latitude', 'longitude']].drop_duplicates()
                    weather_data = self.gfs_fetcher.fetch_rainfall_forecast(stations)
                except Exception as e:
                    logger.warning(f"Weather data fetch failed: {e}")
                    weather_data = None
            
            predictions = self.generate_predictions(features, weather_data=weather_data)
            
            # Get station metadata
            station_metadata = cleaned_data[['station_id', 'station_name', 'river_name', 'basin', 'state', 'latitude', 'longitude', 'district']].drop_duplicates()
            
            # Update statistics
            cycle_stats.total_stations = len(predictions)
            cycle_stats.high_risk_count = len(predictions[predictions['risk_tier_name'].str.lower() == 'high'])
            cycle_stats.medium_risk_count = len(predictions[predictions['risk_tier_name'].str.lower() == 'medium'])
            cycle_stats.low_risk_count = len(predictions[predictions['risk_tier_name'].str.lower() == 'low'])
            
            # Log prediction mode
            if self.mode == "WEATHER-AWARE" and 'prediction_mode' in predictions.columns:
                actual_mode = predictions['prediction_mode'].iloc[0]
                logger.info(f"Actual prediction mode: {actual_mode}")
                if actual_mode == "WEATHER-AWARE":
                    logger.info("✅ Using GFS weather data")
                else:
                    logger.info("⚠️ Fell back to HYDRO-ONLY (weather data unavailable)")
            
            # Generate JSON predictions
            json_stats = self.output_manager.save_all_predictions_json(
                predictions,
                station_metadata
            )
            
            cycle_stats.total_json_generated = json_stats.get('json_generated', 0)
            cycle_stats.low_risk_json_generated = json_stats.get('low_risk_json_generated', 0)
            
            # Process alerts
            alert_stats = self.alert_manager.process_predictions(
                predictions,
                station_metadata
            )
            
            cycle_stats.alerts_triggered = alert_stats.get('alerts_triggered', 0)
            
            # Print cycle summary
            cycle_stats.print_summary()
            
            logger.info("="*70)
            logger.info(f"CYCLE {cycle_number} COMPLETE - MODE: {self.mode}")
            logger.info("="*70 + "\n")
            
            return {
                'status': 'success',
                'cycle_number': cycle_number,
                'mode': self.mode,
                'statistics': cycle_stats,
            }
        
        except Exception as e:
            logger.error(f"Cycle {cycle_number} failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'cycle_number': cycle_number,
                'mode': self.mode,
                'error': str(e),
            }


def main():
    """Main entry point for autonomous operation."""
    parser = argparse.ArgumentParser(
        description="Autonomous Flood Early Warning Prediction System"
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=30,
        help='Execution interval in minutes (default: 30)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['weather-aware', 'hydro-only'],
        default='weather-aware',
        help='Prediction mode: weather-aware (GFS + hydro) or hydro-only (default: weather-aware)'
    )
    
    args = parser.parse_args()
    
    # Determine if weather-aware mode
    use_weather_aware = (args.mode == 'weather-aware')
    
    logger.info("="*70)
    logger.info("AUTONOMOUS FLOOD EARLY WARNING PREDICTION SYSTEM")
    logger.info("="*70)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Interval: {args.interval} minutes")
    logger.info("="*70 + "\n")
    
    # Initialize pipeline and scheduler
    pipeline = AutonomousFloodPredictionPipeline(use_weather_aware=use_weather_aware)
    scheduler = AutonomousScheduler(interval_minutes=args.interval)
    
    # Start autonomous operation
    scheduler.start(pipeline_func=pipeline.run_cycle)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Autonomous system stopped by user")
        sys.exit(0)

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
from src.utils.constants import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_NAMES
from src.utils.output_manager import OutputManager
from src.utils.scheduler import AutonomousScheduler, CycleStatistics
from src.utils.alert_manager import AlertManager
from src.data_engineering import DataCleaner
from src.features import FeatureGenerator
from src.models import ModelTrainer
from src.inference import FloodPredictor, AlertGenerator
from src.explainability import ExplanationGenerator

logger = setup_logger(__name__)


class AutonomousFloodPredictionPipeline:
    """
    Autonomous flood prediction pipeline for continuous operation.
    """
    
    def __init__(self):
        """Initialize pipeline."""
        self.cleaner = DataCleaner()
        self.feature_gen = FeatureGenerator()
        self.trainer = ModelTrainer()
        self.predictor = FloodPredictor()
        self.alert_gen = AlertGenerator()
        self.explainer = ExplanationGenerator()
        self.output_manager = OutputManager()
        self.alert_manager = AlertManager()
        logger.info("AutonomousFloodPredictionPipeline initialized")
    
    def load_raw_data(self, csv_path: str) -> pd.DataFrame:
        """Load raw CWC data from CSV."""
        df = pd.read_csv(csv_path)
        return df
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        cleaned = self.cleaner.clean(raw_data)
        return cleaned
    
    def generate_features(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML features."""
        data_dict = {
            'stations': cleaned_data[['station_id', 'station_name', 'river_name', 'basin', 'state']].drop_duplicates(),
            'levels': cleaned_data
        }
        
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
    
    def generate_predictions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using trained models."""
        self.predictor = FloodPredictor(model_dir=str(MODELS_DIR))
        predictions = self.predictor.predict_all(features)
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
            
            # Generate predictions
            predictions = self.generate_predictions(features)
            
            # Get station metadata (include latitude, longitude, district)
            station_metadata = cleaned_data[['station_id', 'station_name', 'river_name', 'basin', 'state', 'latitude', 'longitude', 'district']].drop_duplicates()
            
            # Update statistics
            cycle_stats.total_stations = len(predictions)
            cycle_stats.high_risk_count = len(predictions[predictions['risk_tier_name'].str.lower() == 'high'])
            cycle_stats.medium_risk_count = len(predictions[predictions['risk_tier_name'].str.lower() == 'medium'])
            cycle_stats.low_risk_count = len(predictions[predictions['risk_tier_name'].str.lower() == 'low'])
            
            # Generate JSON predictions (CHANGE 3: Updated rules)
            json_stats = self.output_manager.save_all_predictions_json(
                predictions,
                station_metadata
            )
            
            cycle_stats.total_json_generated = json_stats.get('json_generated', 0)
            cycle_stats.low_risk_json_generated = json_stats.get('low_risk_json_generated', 0)
            
            # Process alerts (CHANGE 2: Alert behavior)
            alert_stats = self.alert_manager.process_predictions(
                predictions,
                station_metadata
            )
            
            cycle_stats.alerts_triggered = alert_stats.get('alerts_triggered', 0)
            
            # Print cycle summary (CHANGE 5: Execution logging)
            cycle_stats.print_summary()
            
            return {
                'status': 'success',
                'cycle_number': cycle_number,
                'statistics': cycle_stats,
            }
        
        except Exception as e:
            logger.error(f"Cycle {cycle_number} failed: {str(e)}", exc_info=True)
            return {
                'status': 'failed',
                'cycle_number': cycle_number,
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
    
    args = parser.parse_args()
    
    logger.info(f"Starting autonomous mode with {args.interval}-minute interval")
    
    # Initialize pipeline and scheduler
    pipeline = AutonomousFloodPredictionPipeline()
    scheduler = AutonomousScheduler(interval_minutes=args.interval)
    
    # Start autonomous operation
    scheduler.start(pipeline_func=pipeline.run_cycle)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Autonomous system stopped by user")
        sys.exit(0)

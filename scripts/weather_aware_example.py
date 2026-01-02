"""
Example: Weather-Aware Flood Prediction System

Demonstrates:
1. Fetching GFS rainfall forecasts
2. Generating weather-aware features
3. Making multi-horizon predictions
4. Handling weather data gracefully

BACKWARD-COMPATIBLE: Falls back to hydro-only if weather unavailable.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import weather modules
from src.weather.gfs_fetcher import GFSFetcher
from src.weather.weather_cache import WeatherCache
from src.features.weather_aware_feature_generator import WeatherAwareFeatureGenerator
from src.inference.weather_aware_predictor import WeatherAwarePredictor
from src.utils.output_manager import OutputManager
from src.utils.weather_output_manager import WeatherAwareOutputManager

# Import existing modules (unchanged)
from src.data_engineering import DataCleaner
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def example_weather_aware_pipeline():
    """
    Complete weather-aware flood prediction pipeline.
    
    Demonstrates:
    - Graceful weather data fetching
    - Feature generation with weather data
    - Multi-horizon predictions
    - Weather-aware JSON output
    """
    
    logger.info("="*70)
    logger.info("WEATHER-AWARE FLOOD PREDICTION SYSTEM - EXAMPLE")
    logger.info("="*70)
    
    # ========================================================================
    # STEP 1: Load and clean data (existing code, unchanged)
    # ========================================================================
    
    logger.info("\nSTEP 1: Loading and cleaning hydrological data...")
    
    # Load raw data (example)
    raw_data_file = Path("data/raw/cwc_raw_20260102_0804.csv")
    if not raw_data_file.exists():
        logger.error(f"Data file not found: {raw_data_file}")
        return
    
    raw_data = pd.read_csv(raw_data_file)
    logger.info(f"Loaded {len(raw_data)} raw records")
    
    # Clean data (existing code)
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(raw_data)
    logger.info(f"Cleaned data: {len(cleaned_data)} records")
    
    # ========================================================================
    # STEP 2: Fetch weather data (NEW - with fail-safe)
    # ========================================================================
    
    logger.info("\nSTEP 2: Fetching GFS rainfall forecasts...")
    
    weather_data = None
    try:
        # Initialize GFS fetcher with caching
        gfs = GFSFetcher(cache_dir=Path("data/cache"))
        
        # Get unique stations
        stations = cleaned_data[['station_id', 'latitude', 'longitude']].drop_duplicates()
        logger.info(f"Fetching forecasts for {len(stations)} stations...")
        
        # Fetch with timeout (fail-safe)
        weather_data = gfs.fetch_rainfall_forecast(
            stations,
            forecast_hours=24
        )
        
        if weather_data is not None and not weather_data.empty:
            logger.info(f"Successfully fetched weather data for {len(weather_data)} stations")
            logger.info(f"Weather data columns: {list(weather_data.columns)}")
        else:
            logger.warning("Weather fetch returned empty data")
            weather_data = None
    
    except Exception as e:
        logger.warning(f"Weather fetch failed: {str(e)}")
        logger.warning("Continuing in hydro-only mode...")
        weather_data = None
    
    # ========================================================================
    # STEP 3: Generate weather-aware features
    # ========================================================================
    
    logger.info("\nSTEP 3: Generating weather-aware features...")
    
    feature_gen = WeatherAwareFeatureGenerator()
    features = feature_gen.generate(
        cleaned_data,
        weather_data=weather_data,
        historical_rainfall_stats=None
    )
    
    logger.info(f"Generated {len(features)} feature records")
    logger.info(f"Feature columns: {len(features.columns)}")
    
    # Log feature availability
    hydro_features = feature_gen.get_hydro_feature_names()
    weather_features = feature_gen.get_weather_feature_names()
    
    logger.info(f"Hydrological features: {len(hydro_features)}")
    logger.info(f"Weather features: {len(weather_features)}")
    
    # ========================================================================
    # STEP 4: Make weather-aware predictions
    # ========================================================================
    
    logger.info("\nSTEP 4: Making weather-aware predictions...")
    
    predictor = WeatherAwarePredictor()
    predictions = predictor.predict_with_weather(
        features,
        weather_data=weather_data
    )
    
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Log prediction mode
    if predictions.empty:
        logger.warning("No predictions generated")
        return
    
    mode = predictions['prediction_mode'].iloc[0]
    logger.info(f"Prediction mode: {mode}")
    
    # ========================================================================
    # STEP 5: Analyze predictions
    # ========================================================================
    
    logger.info("\nSTEP 5: Analyzing predictions...")
    
    # Risk distribution
    risk_dist = predictions['risk_tier_name'].value_counts()
    logger.info("Risk distribution (now):")
    for risk_level, count in risk_dist.items():
        logger.info(f"  {risk_level}: {count} stations")
    
    # Multi-horizon risk
    if 'risk_tier_6h' in predictions.columns:
        risk_dist_6h = predictions['risk_tier_6h'].value_counts()
        logger.info("Risk distribution (+6h):")
        for risk_level, count in risk_dist_6h.items():
            logger.info(f"  {risk_level}: {count} stations")
    
    if 'risk_tier_24h' in predictions.columns:
        risk_dist_24h = predictions['risk_tier_24h'].value_counts()
        logger.info("Risk distribution (+24h):")
        for risk_level, count in risk_dist_24h.items():
            logger.info(f"  {risk_level}: {count} stations")
    
    # ========================================================================
    # STEP 6: Generate weather-aware JSON output
    # ========================================================================
    
    logger.info("\nSTEP 6: Generating weather-aware JSON output...")
    
    output_manager = OutputManager()
    
    # Select HIGH and MEDIUM risk stations
    high_risk = predictions[predictions['risk_tier_name'] == 'HIGH']
    medium_risk = predictions[predictions['risk_tier_name'] == 'MEDIUM']
    
    logger.info(f"HIGH risk stations: {len(high_risk)}")
    logger.info(f"MEDIUM risk stations: {len(medium_risk)}")
    
    # Generate JSON for HIGH and MEDIUM risk
    json_count = 0
    for idx, row in pd.concat([high_risk, medium_risk]).iterrows():
        try:
            station_id = row['station_id']
            
            # Build base prediction JSON
            prediction_json = {
                'prediction': {
                    'flood_probability': round(float(row['risk_score']), 4),
                    'risk_level': row['risk_tier_name'],
                },
                'confidence': {
                    'model_confidence': 0.85,
                },
                'explanation': [
                    f"Risk score: {row['risk_score']:.1%}",
                ],
                'context': {
                    'location': {
                        'district': row.get('district', 'Unknown'),
                        'state': row.get('state', 'Unknown'),
                        'latitude': float(row.get('latitude', 0)),
                        'longitude': float(row.get('longitude', 0)),
                    },
                    'time_window': {
                        'prediction_time': datetime.utcnow().isoformat() + 'Z',
                        'valid_until': (datetime.utcnow() + pd.Timedelta(hours=6)).isoformat() + 'Z',
                    }
                },
                'metadata': {
                    'model_type': 'ml_hybrid',
                    'model_version': '1.0',
                    'data_source': ['Central Water Commission (CWC), Government of India'],
                }
            }
            
            # Extend with weather data
            prediction_json = WeatherAwareOutputManager.extend_prediction_json_with_weather(
                prediction_json,
                row
            )
            
            # Add weather-based explanation
            weather_explanation = WeatherAwareOutputManager.generate_weather_explanation(
                row,
                row['risk_tier_name']
            )
            if weather_explanation:
                prediction_json['explanation'].append(weather_explanation)
            
            # Add predictive alert if future risk is HIGH
            prediction_json = WeatherAwareOutputManager.add_predictive_alert_flag(
                prediction_json,
                row
            )
            
            # Save JSON
            output_manager.save_prediction_json(station_id, prediction_json)
            json_count += 1
        
        except Exception as e:
            logger.error(f"Error generating JSON for {station_id}: {str(e)}")
            continue
    
    logger.info(f"Generated {json_count} JSON files")
    
    # ========================================================================
    # STEP 7: Summary
    # ========================================================================
    
    logger.info("\n" + "="*70)
    logger.info("WEATHER-AWARE PIPELINE COMPLETE")
    logger.info("="*70)
    logger.info(f"Prediction mode: {mode}")
    logger.info(f"Weather data used: {predictions['weather_data_used'].iloc[0]}")
    logger.info(f"Total predictions: {len(predictions)}")
    logger.info(f"HIGH risk: {len(high_risk)}")
    logger.info(f"MEDIUM risk: {len(medium_risk)}")
    logger.info(f"JSON files generated: {json_count}")
    logger.info(f"JSON location: data/outputs/json_predictions/")
    logger.info("="*70 + "\n")


def example_hydro_only_mode():
    """
    Example: Hydro-only mode (backward compatible).
    
    Shows that existing code works unchanged.
    """
    
    logger.info("="*70)
    logger.info("HYDRO-ONLY MODE - BACKWARD COMPATIBLE")
    logger.info("="*70)
    
    # This is the existing code - works unchanged
    from src.features.feature_generator import FeatureGenerator
    from src.inference.predictor import FloodPredictor
    
    # Load data
    raw_data_file = Path("data/raw/cwc_raw_20260102_0804.csv")
    if not raw_data_file.exists():
        logger.error(f"Data file not found: {raw_data_file}")
        return
    
    raw_data = pd.read_csv(raw_data_file)
    
    # Clean data
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(raw_data)
    
    # Generate hydro features (existing code)
    feature_gen = FeatureGenerator()
    features = feature_gen.generate(cleaned_data)
    
    # Make predictions (existing code)
    predictor = FloodPredictor()
    predictions = predictor.predict_all(features)
    
    logger.info(f"Generated {len(predictions)} predictions in hydro-only mode")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    # Run weather-aware example
    example_weather_aware_pipeline()
    
    # Show that hydro-only mode still works
    # example_hydro_only_mode()

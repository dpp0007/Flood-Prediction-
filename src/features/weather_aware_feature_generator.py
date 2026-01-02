"""
Weather-Aware Feature Generator.

Extends the base feature generator with optional weather features.
BACKWARD-COMPATIBLE: Works with or without weather data.

If weather data is available:
- Adds 7 new weather-derived features
- Keeps all existing hydro features unchanged
- Enables weather-aware predictions

If weather data is unavailable:
- Falls back to hydro-only mode
- All weather features default to 0
- System works exactly as before
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

from src.features.feature_generator import FeatureGenerator
from src.weather.weather_features import WeatherFeatureGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WeatherAwareFeatureGenerator:
    """
    Generates both hydro and weather-aware features.
    
    Combines:
    - Base hydrological features (10 features)
    - Weather-derived features (7 features)
    - Total: 17 features
    
    BACKWARD-COMPATIBLE:
    - If weather data missing: uses 10 hydro features only
    - Existing models can use subset of features
    - New models can use full feature set
    """
    
    def __init__(self, lookback_hours: int = 24, rate_window_hours: int = 3):
        """
        Initialize weather-aware feature generator.
        
        Args:
            lookback_hours: Historical window for feature calculation
            rate_window_hours: Window for rate of rise calculation
        """
        self.hydro_generator = FeatureGenerator(
            lookback_hours=lookback_hours,
            rate_window_hours=rate_window_hours
        )
        self.weather_generator = WeatherFeatureGenerator()
        
        logger.info("Weather-aware feature generator initialized")
    
    def generate(self, cleaned_data: Dict[str, pd.DataFrame],
                weather_data: Optional[pd.DataFrame] = None,
                historical_rainfall_stats: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate hydro + weather features.
        
        Args:
            cleaned_data: Cleaned hydrological data
            weather_data: Optional GFS/IMD weather data
            historical_rainfall_stats: Optional historical rainfall statistics
            
        Returns:
            DataFrame with all features (hydro + weather)
        """
        logger.info("Generating weather-aware features...")
        
        # Step 1: Generate base hydrological features
        features_df = self.hydro_generator.generate(cleaned_data)
        logger.info(f"Generated {len(features_df)} hydro features")
        
        # Step 2: Add weather features (optional)
        if weather_data is not None and not weather_data.empty:
            logger.info(f"Adding weather features from {len(weather_data)} stations...")
            features_df = self.weather_generator.add_weather_features(
                features_df,
                weather_data=weather_data,
                historical_rainfall_stats=historical_rainfall_stats
            )
            logger.info("Weather features added successfully")
        else:
            logger.info("No weather data available, adding zero-valued weather features...")
            features_df = self.weather_generator.add_weather_features(
                features_df,
                weather_data=None,
                historical_rainfall_stats=None
            )
            logger.info("Weather features initialized to zero (hydro-only mode)")
        
        # Step 3: Log feature availability
        self._log_feature_availability(features_df, weather_data)
        
        return features_df
    
    def get_all_feature_names(self) -> list:
        """
        Get all feature names (hydro + weather).
        
        Returns:
            List of all feature column names
        """
        from src.utils.constants import FEATURE_NAMES
        
        hydro_features = FEATURE_NAMES
        weather_features = self.weather_generator.get_weather_feature_names()
        
        return hydro_features + weather_features
    
    def get_hydro_feature_names(self) -> list:
        """
        Get hydrological feature names only.
        
        Returns:
            List of hydro feature column names
        """
        from src.utils.constants import FEATURE_NAMES
        return FEATURE_NAMES
    
    def get_weather_feature_names(self) -> list:
        """
        Get weather feature names only.
        
        Returns:
            List of weather feature column names
        """
        return self.weather_generator.get_weather_feature_names()
    
    def _log_feature_availability(self, features_df: pd.DataFrame,
                                 weather_data: Optional[pd.DataFrame]) -> None:
        """
        Log feature availability for transparency.
        
        Args:
            features_df: Feature DataFrame
            weather_data: Weather data (if available)
        """
        from src.utils.constants import FEATURE_NAMES
        
        # Count available features
        hydro_features_available = sum(
            1 for col in FEATURE_NAMES if col in features_df.columns
        )
        weather_features_available = sum(
            1 for col in self.weather_generator.get_weather_feature_names()
            if col in features_df.columns
        )
        
        # Check if weather features are non-zero
        weather_features_active = False
        if weather_data is not None and not weather_data.empty:
            weather_features_active = True
        
        logger.info(f"Feature availability:")
        logger.info(f"  Hydrological features: {hydro_features_available}/{len(FEATURE_NAMES)}")
        logger.info(f"  Weather features: {weather_features_available}/7")
        logger.info(f"  Weather data active: {weather_features_active}")
        logger.info(f"  Total features: {len(features_df.columns)}")
        
        # Log mode
        if weather_features_active:
            logger.info("MODE: WEATHER-AWARE (hydro + weather)")
        else:
            logger.info("MODE: HYDRO-ONLY (weather features zero-valued)")

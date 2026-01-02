"""
Weather-Derived Feature Engineering for Flood Prediction.

Generates weather-aware features from GFS forecasts and IMD observations:
- Rainfall-based features
- Hydro-weather interaction features
- Compound risk indices

All features are OPTIONAL and BACKWARD-COMPATIBLE:
- If weather data is missing: features default to 0
- Existing hydro features are NOT modified
- System works in hydro-only mode if weather data unavailable
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class WeatherFeatureGenerator:
    """
    Generates weather-aware features from rainfall forecasts.
    
    Features added:
    - rainfall_last_6h_mm: Observed rainfall (last 6 hours)
    - rainfall_forecast_next_6h_mm: GFS forecast (next 6 hours)
    - rainfall_forecast_next_24h_mm: GFS forecast (next 24 hours)
    - rainfall_intensity_trend: Rising/falling trend
    - rainfall_runoff_index: Rainfall × station volatility
    - saturation_proxy: Rainfall / historical average
    - compound_risk_index: Rainfall × distance to danger
    """
    
    def __init__(self):
        """Initialize weather feature generator."""
        logger.info("Weather feature generator initialized")
    
    def add_weather_features(self, features_df: pd.DataFrame,
                            weather_data: Optional[pd.DataFrame] = None,
                            historical_rainfall_stats: Optional[Dict] = None) -> pd.DataFrame:
        """
        Add weather-derived features to feature DataFrame.
        
        BACKWARD-COMPATIBLE: If weather_data is None, features default to 0.
        
        Args:
            features_df: Existing feature DataFrame (with hydro features)
            weather_data: DataFrame with rainfall forecasts from GFS
            historical_rainfall_stats: Dictionary with historical rainfall statistics
            
        Returns:
            DataFrame with added weather features
        """
        features_df = features_df.copy()
        
        logger.info(f"Adding weather features to {len(features_df)} records...")
        
        # Initialize weather features with zeros (fail-safe)
        features_df['rainfall_last_6h_mm'] = 0.0
        features_df['rainfall_forecast_next_6h_mm'] = 0.0
        features_df['rainfall_forecast_next_24h_mm'] = 0.0
        features_df['rainfall_intensity_trend'] = 0.0
        features_df['rainfall_runoff_index'] = 0.0
        features_df['saturation_proxy'] = 0.0
        features_df['compound_risk_index'] = 0.0
        
        # If weather data is available, populate features
        if weather_data is not None and not weather_data.empty:
            logger.info(f"Merging weather data for {len(weather_data)} stations...")
            
            # Merge weather data
            features_df = features_df.merge(
                weather_data[['station_id', 'rainfall_last_6h_mm',
                             'rainfall_forecast_next_6h_mm',
                             'rainfall_forecast_next_24h_mm']],
                on='station_id',
                how='left',
                suffixes=('', '_weather')
            )
            
            # Update rainfall columns from weather data
            for col in ['rainfall_last_6h_mm', 'rainfall_forecast_next_6h_mm',
                       'rainfall_forecast_next_24h_mm']:
                weather_col = f"{col}_weather"
                if weather_col in features_df.columns:
                    features_df[col] = features_df[weather_col].fillna(features_df[col])
                    features_df.drop(columns=[weather_col], inplace=True)
            
            logger.info(f"Weather data merged for {len(weather_data)} stations")
        
        # Generate derived features
        features_df = self._generate_derived_features(
            features_df,
            historical_rainfall_stats
        )
        
        # Normalize all weather features to [0, 1]
        features_df = self._normalize_weather_features(features_df)
        
        logger.info("Weather features added successfully")
        
        return features_df
    
    def _generate_derived_features(self, features_df: pd.DataFrame,
                                  historical_rainfall_stats: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate derived weather features.
        
        Args:
            features_df: Feature DataFrame
            historical_rainfall_stats: Historical rainfall statistics
            
        Returns:
            DataFrame with derived features
        """
        # Rainfall intensity trend (rising/falling)
        # Positive if forecast > observed, negative if forecast < observed
        features_df['rainfall_intensity_trend'] = (
            features_df['rainfall_forecast_next_6h_mm'] - 
            features_df['rainfall_last_6h_mm']
        )
        
        # Rainfall runoff index
        # Combines forecast rainfall with station volatility
        # Higher volatility = more runoff from same rainfall
        if 'station_volatility' in features_df.columns:
            features_df['rainfall_runoff_index'] = (
                features_df['rainfall_forecast_next_6h_mm'] * 
                features_df['station_volatility']
            )
        else:
            features_df['rainfall_runoff_index'] = (
                features_df['rainfall_forecast_next_6h_mm'] * 0.5  # Default volatility
            )
        
        # Saturation proxy
        # Ratio of recent rainfall to historical average
        # High ratio = basin is saturated, more runoff expected
        if historical_rainfall_stats and 'mean_rainfall_mm' in historical_rainfall_stats:
            hist_mean = historical_rainfall_stats['mean_rainfall_mm']
            if hist_mean > 0:
                features_df['saturation_proxy'] = (
                    features_df['rainfall_last_6h_mm'] / hist_mean
                )
            else:
                features_df['saturation_proxy'] = 0.0
        else:
            # Default: assume historical mean of 5mm
            features_df['saturation_proxy'] = (
                features_df['rainfall_last_6h_mm'] / 5.0
            )
        
        # Compound risk index
        # Combines forecast rainfall with proximity to danger level
        # High rainfall + close to danger = high compound risk
        if 'distance_to_danger' in features_df.columns:
            # Normalize distance to [0, 1]
            max_distance = features_df['distance_to_danger'].max()
            if max_distance > 0:
                distance_norm = 1.0 - (features_df['distance_to_danger'] / max_distance)
            else:
                distance_norm = 0.5
            
            features_df['compound_risk_index'] = (
                features_df['rainfall_forecast_next_24h_mm'] * distance_norm
            )
        else:
            features_df['compound_risk_index'] = (
                features_df['rainfall_forecast_next_24h_mm'] * 0.5
            )
        
        return features_df
    
    def _normalize_weather_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize weather features to [0, 1] range.
        
        Args:
            features_df: Feature DataFrame
            
        Returns:
            DataFrame with normalized features
        """
        weather_feature_cols = [
            'rainfall_last_6h_mm',
            'rainfall_forecast_next_6h_mm',
            'rainfall_forecast_next_24h_mm',
            'rainfall_intensity_trend',
            'rainfall_runoff_index',
            'saturation_proxy',
            'compound_risk_index',
        ]
        
        for col in weather_feature_cols:
            if col in features_df.columns:
                # Get min and max
                col_min = features_df[col].min()
                col_max = features_df[col].max()
                
                # Normalize - but keep raw values if range is small
                if col_max > col_min and (col_max - col_min) > 0.01:
                    # Normal normalization
                    features_df[col] = (features_df[col] - col_min) / (col_max - col_min)
                    features_df[col] = np.clip(features_df[col], 0, 1)
                elif col_max > 0:
                    # If range is small but max > 0, use max normalization
                    features_df[col] = features_df[col] / col_max
                    features_df[col] = np.clip(features_df[col], 0, 1)
                else:
                    # All zeros - keep as is
                    features_df[col] = 0.0
        
        return features_df
    
    def get_weather_feature_names(self) -> list:
        """
        Get list of weather feature names.
        
        Returns:
            List of weather feature column names
        """
        return [
            'rainfall_last_6h_mm',
            'rainfall_forecast_next_6h_mm',
            'rainfall_forecast_next_24h_mm',
            'rainfall_intensity_trend',
            'rainfall_runoff_index',
            'saturation_proxy',
            'compound_risk_index',
        ]
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of weather features.
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'rainfall_last_6h_mm': 'Observed rainfall in last 6 hours (mm)',
            'rainfall_forecast_next_6h_mm': 'GFS forecast rainfall for next 6 hours (mm)',
            'rainfall_forecast_next_24h_mm': 'GFS forecast rainfall for next 24 hours (mm)',
            'rainfall_intensity_trend': 'Trend in rainfall intensity (forecast - observed)',
            'rainfall_runoff_index': 'Rainfall × station volatility (runoff potential)',
            'saturation_proxy': 'Recent rainfall / historical average (basin saturation)',
            'compound_risk_index': 'Forecast rainfall × proximity to danger level',
        }

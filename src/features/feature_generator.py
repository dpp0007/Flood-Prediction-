"""
Feature Engineering for Flood Risk Prediction.

Data Source: Central Water Commission (CWC), Government of India

Generates predictive features from hydrological data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from datetime import datetime, timedelta
from src.utils.logger import setup_logger
from src.utils.constants import FEATURE_NAMES

logger = setup_logger(__name__)

class FeatureGenerator:
    """
    Generates ML features from cleaned hydrological data.
    
    Features:
    - Current water level
    - Distance to warning/danger levels
    - Rate of rise (Δ level/hour)
    - Consecutive rising hours
    - Historical percentiles
    - Basin-level aggregations
    - Station volatility
    """
    
    def __init__(self, lookback_hours: int = 24, rate_window_hours: int = 3):
        """
        Initialize feature generator.
        
        Args:
            lookback_hours: Historical window for feature calculation
            rate_window_hours: Window for rate of rise calculation
        """
        self.lookback_hours = lookback_hours
        self.rate_window_hours = rate_window_hours
        logger.info(f"Feature generator initialized (lookback={lookback_hours}h, rate_window={rate_window_hours}h)")
    
    def generate(self, cleaned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate all features from cleaned data.
        
        Args:
            cleaned_data: Dictionary with cleaned DataFrames or a single DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Generating features...")
        
        # Handle both dict and DataFrame inputs
        if isinstance(cleaned_data, pd.DataFrame):
            levels = cleaned_data.copy()
            stations = levels[['station_id', 'station_name', 'river_name', 'basin', 'state']].drop_duplicates()
        else:
            stations = cleaned_data.get('stations', pd.DataFrame())
            levels = cleaned_data.get('levels', cleaned_data)
        
        # Merge station metadata with current levels
        features_df = levels.merge(stations, on='station_id', how='left') if not stations.empty else levels.copy()
        
        # Ensure timestamp column exists
        if 'timestamp' in features_df.columns and 'observation_time' not in features_df.columns:
            features_df['observation_time'] = features_df['timestamp']
        
        # Generate features
        features_df['current_level'] = features_df['current_level']
        features_df['distance_to_warning'] = features_df['warning_level'] - features_df['current_level']
        features_df['distance_to_danger'] = features_df['danger_level'] - features_df['current_level']
        
        # Rate of rise (simulated from trend)
        features_df['rate_of_rise_3h'] = features_df.apply(
            lambda row: self._calculate_rate_of_rise(row, levels),
            axis=1
        )
        
        # Consecutive rising hours
        features_df['consecutive_rising_hours'] = features_df.apply(
            lambda row: self._calculate_consecutive_rising(row),
            axis=1
        )
        
        # Historical percentile
        features_df['historical_percentile'] = features_df.apply(
            lambda row: self._calculate_historical_percentile(row, levels),
            axis=1
        )
        
        # Basin-level aggregation - handle missing basin column
        if 'basin' in features_df.columns:
            features_df['basin_avg_level'] = features_df.groupby('basin')['current_level'].transform('mean')
        else:
            features_df['basin_avg_level'] = features_df['current_level'].rolling(window=3, min_periods=1).mean()
        
        # Station volatility
        features_df['station_volatility'] = features_df.apply(
            lambda row: self._calculate_volatility(row, levels),
            axis=1
        )
        
        # Time-based features
        features_df['observation_time'] = pd.to_datetime(features_df['observation_time'], errors='coerce')
        
        # Ensure naive datetime (remove timezone)
        if hasattr(features_df['observation_time'].dtype, 'tz') and features_df['observation_time'].dtype.tz is not None:
            features_df['observation_time'] = features_df['observation_time'].dt.tz_localize(None)
        
        # Calculate hours since last update
        try:
            current_time = pd.Timestamp.utcnow()
            time_diff = current_time - features_df['observation_time']
            features_df['hours_since_last_update'] = time_diff.dt.total_seconds() / 3600
        except:
            # Fallback if timezone issues persist
            features_df['hours_since_last_update'] = 0.0
        
        features_df['hours_since_last_update'] = features_df['hours_since_last_update'].fillna(0).clip(lower=0)
        
        # Seasonal factor (simplified)
        features_df['month'] = features_df['observation_time'].dt.month
        features_df['seasonal_factor'] = np.sin(2 * np.pi * features_df['month'] / 12)
        
        # Select final features
        feature_cols = [col for col in FEATURE_NAMES if col in features_df.columns]
        features_df = features_df[['station_id', 'observation_time'] + feature_cols]
        
        logger.info(f"Generated features for {len(features_df)} records")
        logger.info(f"Feature columns: {feature_cols}")
        
        return features_df
    
    def _calculate_rate_of_rise(self, row: pd.Series, levels_df: pd.DataFrame) -> float:
        """Calculate rate of rise in meters per hour."""
        station_id = row['station_id']
        current_time = row.get('observation_time', row.get('timestamp'))
        
        # Get historical data for this station
        station_data = levels_df[levels_df['station_id'] == station_id].copy()
        
        # Ensure timestamp column exists
        if 'timestamp' in station_data.columns and 'observation_time' not in station_data.columns:
            station_data['observation_time'] = station_data['timestamp']
        
        station_data['observation_time'] = pd.to_datetime(station_data['observation_time'], errors='coerce')
        station_data = station_data.sort_values('observation_time')
        
        # Find data from N hours ago
        current_time = pd.to_datetime(current_time, errors='coerce')
        cutoff_time = current_time - timedelta(hours=self.rate_window_hours)
        historical = station_data[station_data['observation_time'] <= cutoff_time]
        
        if len(historical) == 0:
            return 0.0
        
        old_level = historical.iloc[-1]['current_level']
        current_level = row['current_level']
        
        rate = (current_level - old_level) / self.rate_window_hours
        return max(0, rate)  # Only positive rates (rising)
    
    def _calculate_consecutive_rising(self, row: pd.Series) -> int:
        """Calculate consecutive hours of rising water level."""
        trend = row.get('trend', 'unknown')
        
        if trend == 'rising':
            return 3  # Simplified: assume 3 hours if rising
        elif trend == 'falling':
            return 0
        else:
            return 1  # Neutral
    
    def _calculate_historical_percentile(self, row: pd.Series, levels_df: pd.DataFrame) -> float:
        """Calculate percentile of current level vs historical data."""
        station_id = row['station_id']
        current_level = row['current_level']
        
        # Get historical data for this station
        station_data = levels_df[levels_df['station_id'] == station_id]['current_level']
        
        if len(station_data) < 2:
            return 50.0
        
        percentile = (station_data < current_level).sum() / len(station_data) * 100
        return percentile
    
    def _calculate_volatility(self, row: pd.Series, levels_df: pd.DataFrame) -> float:
        """Calculate station-level volatility (std dev of recent levels)."""
        station_id = row['station_id']
        
        # Get recent data for this station
        station_data = levels_df[levels_df['station_id'] == station_id]['current_level']
        
        if len(station_data) < 2:
            return 0.0
        
        volatility = station_data.std()
        return volatility if not np.isnan(volatility) else 0.0

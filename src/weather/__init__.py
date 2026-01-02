"""
Weather Data Integration Module for Flood Prediction System.

Provides:
- GFS rainfall forecast fetching
- IMD historical rainfall (where legally available)
- Weather feature engineering
- Fail-safe weather data handling

This module is OPTIONAL and BACKWARD-COMPATIBLE.
If weather data is unavailable, the system falls back to hydro-only mode.
"""

from src.weather.gfs_fetcher import GFSFetcher
from src.weather.weather_features import WeatherFeatureGenerator
from src.weather.weather_cache import WeatherCache

__all__ = [
    'GFSFetcher',
    'WeatherFeatureGenerator',
    'WeatherCache',
]

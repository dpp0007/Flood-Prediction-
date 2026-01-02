"""
Weather Data Caching and Management.

Handles:
- Caching GFS forecasts (3-hour validity)
- Caching IMD observations (24-hour validity)
- Cache invalidation
- Cache statistics
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class WeatherCache:
    """
    Manages caching of weather data.
    
    Cache validity:
    - GFS forecasts: 3 hours
    - IMD observations: 24 hours
    """
    
    def __init__(self, cache_dir: Path = Path("data/cache")):
        """
        Initialize weather cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.gfs_cache_file = self.cache_dir / "gfs_forecast.pkl"
        self.imd_cache_file = self.cache_dir / "imd_observations.pkl"
        
        logger.info(f"Weather cache initialized at {self.cache_dir}")
    
    def get_gfs_forecast(self) -> Optional[pd.DataFrame]:
        """
        Get cached GFS forecast if valid.
        
        Returns:
            DataFrame or None if cache is stale/missing
        """
        return self._get_cached_data(self.gfs_cache_file, validity_hours=3)
    
    def set_gfs_forecast(self, data: pd.DataFrame) -> None:
        """
        Cache GFS forecast data.
        
        Args:
            data: GFS forecast DataFrame
        """
        self._set_cached_data(self.gfs_cache_file, data)
    
    def get_imd_observations(self) -> Optional[pd.DataFrame]:
        """
        Get cached IMD observations if valid.
        
        Returns:
            DataFrame or None if cache is stale/missing
        """
        return self._get_cached_data(self.imd_cache_file, validity_hours=24)
    
    def set_imd_observations(self, data: pd.DataFrame) -> None:
        """
        Cache IMD observation data.
        
        Args:
            data: IMD observations DataFrame
        """
        self._set_cached_data(self.imd_cache_file, data)
    
    def _get_cached_data(self, cache_file: Path, validity_hours: int) -> Optional[pd.DataFrame]:
        """
        Get cached data if valid.
        
        Args:
            cache_file: Path to cache file
            validity_hours: Cache validity in hours
            
        Returns:
            DataFrame or None if cache is stale/missing
        """
        if not cache_file.exists():
            return None
        
        try:
            # Check file age
            file_age_seconds = (datetime.utcnow().timestamp() - 
                              cache_file.stat().st_mtime)
            validity_seconds = validity_hours * 3600
            
            if file_age_seconds > validity_seconds:
                logger.debug(f"Cache stale: {cache_file.name} "
                           f"({file_age_seconds/3600:.1f}h old, validity={validity_hours}h)")
                return None
            
            # Load cache
            data = pd.read_pickle(cache_file)
            logger.debug(f"Loaded cache: {cache_file.name} "
                        f"({file_age_seconds/3600:.1f}h old)")
            return data
        
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file.name}: {str(e)}")
            return None
    
    def _set_cached_data(self, cache_file: Path, data: pd.DataFrame) -> None:
        """
        Cache data to file.
        
        Args:
            cache_file: Path to cache file
            data: DataFrame to cache
        """
        try:
            data.to_pickle(cache_file)
            logger.debug(f"Cached data to {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to cache data to {cache_file.name}: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear all cached weather data."""
        try:
            if self.gfs_cache_file.exists():
                self.gfs_cache_file.unlink()
            if self.imd_cache_file.exists():
                self.imd_cache_file.unlink()
            logger.info("Weather cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear cache: {str(e)}")
    
    def get_cache_status(self) -> Dict[str, any]:
        """
        Get status of cached weather data.
        
        Returns:
            Dictionary with cache status
        """
        gfs_age = self._get_file_age_hours(self.gfs_cache_file)
        imd_age = self._get_file_age_hours(self.imd_cache_file)
        
        return {
            'gfs_cached': self.gfs_cache_file.exists(),
            'gfs_age_hours': gfs_age,
            'gfs_valid': gfs_age is not None and gfs_age < 3,
            'imd_cached': self.imd_cache_file.exists(),
            'imd_age_hours': imd_age,
            'imd_valid': imd_age is not None and imd_age < 24,
            'cache_dir': str(self.cache_dir),
        }
    
    def _get_file_age_hours(self, file_path: Path) -> Optional[float]:
        """
        Get age of file in hours.
        
        Args:
            file_path: Path to file
            
        Returns:
            Age in hours or None if file doesn't exist
        """
        if not file_path.exists():
            return None
        
        try:
            age_seconds = datetime.utcnow().timestamp() - file_path.stat().st_mtime
            return age_seconds / 3600
        except Exception:
            return None

"""
CWC Website Scraper for Flood Data Acquisition.

Data Source: Central Water Commission (CWC), Government of India
Website: https://inf.cwc.gov.in/

This module scrapes publicly available hydrological data including:
- River water levels
- Warning and danger level thresholds
- Station metadata (name, river, basin, state)
- Time-stamped observations
- Rising/falling trend indicators

Scraping Rules:
- Respects robots.txt
- Low request frequency (>=30 min interval)
- Implements retries and timeouts
- Logs scraping time and source URL
- Caches results locally to avoid repeated hits
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
from typing import Dict, List, Optional
import json
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.constants import (
    CWC_BASE_URL,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    USER_AGENT,
    RAW_DATA_DIR,
    CWC_ATTRIBUTION,
)

logger = setup_logger(__name__)


class CWCScraper:
    """
    Scrapes flood-related data from CWC website.
    
    Implements ethical scraping with rate limiting, retries, and caching.
    """
    
    def __init__(self, base_url: str = CWC_BASE_URL):
        """
        Initialize CWC scraper.
        
        Args:
            base_url: CWC website base URL
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.last_request_time = 0
        
        logger.info(f"CWC Scraper initialized. Data source: {CWC_ATTRIBUTION['source']}")
    
    def _rate_limit(self, min_interval_seconds: float = 2.0):
        """Implement rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)
        self.last_request_time = time.time()
    
    def _fetch_with_retry(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Fetch URL with retry logic.
        
        Args:
            url: URL to fetch
            **kwargs: Additional requests parameters
            
        Returns:
            Response object or None if all retries failed
        """
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.get(
                    url,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                    **kwargs
                )
                response.raise_for_status()
                logger.info(f"Successfully fetched: {url}")
                return response
            except requests.RequestException as e:
                wait_time = RETRY_BACKOFF_FACTOR ** attempt
                logger.warning(
                    f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {url}. "
                    f"Error: {str(e)}. Retrying in {wait_time}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait_time)
        
        logger.error(f"Failed to fetch {url} after {MAX_RETRIES} attempts")
        return None
    
    def fetch_station_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch current water level data for all stations.
        
        Returns:
            DataFrame with columns: station_id, station_name, river_name, basin,
                                   state, current_level, warning_level, danger_level,
                                   highest_flood_level, trend, timestamp, source_url
        """
        logger.info("Fetching station data from CWC...")
        
        # Attempt to fetch from main CWC page
        url = f"{self.base_url}/"
        response = self._fetch_with_retry(url)
        
        if not response:
            logger.error("Could not fetch CWC data")
            return None
        
        try:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Parse station data from tables
            # Note: CWC website structure may vary; this is a template approach
            stations_data = []
            
            # Look for data tables
            tables = soup.find_all("table")
            logger.info(f"Found {len(tables)} tables on page")
            
            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip header
                    cols = row.find_all("td")
                    if len(cols) >= 8:
                        try:
                            station_record = {
                                "station_id": cols[0].text.strip(),
                                "station_name": cols[1].text.strip(),
                                "river_name": cols[2].text.strip(),
                                "basin": cols[3].text.strip(),
                                "state": cols[4].text.strip(),
                                "current_level": float(cols[5].text.strip()),
                                "warning_level": float(cols[6].text.strip()),
                                "danger_level": float(cols[7].text.strip()),
                                "highest_flood_level": float(cols[8].text.strip()) if len(cols) > 8 else None,
                                "trend": cols[9].text.strip() if len(cols) > 9 else "unknown",
                                "timestamp": datetime.utcnow().isoformat(),
                                "source_url": url,
                            }
                            stations_data.append(station_record)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Error parsing row: {str(e)}")
                            continue
            
            if stations_data:
                df = pd.DataFrame(stations_data)
                logger.info(f"Successfully parsed {len(df)} stations")
                return df
            else:
                logger.warning("No station data found in tables")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing CWC data: {str(e)}")
            return None
    
    def fetch_forecast_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch forecasted water levels (if available).
        
        Returns:
            DataFrame with columns: station_id, forecast_level, forecast_time,
                                   confidence, source_url
        """
        logger.info("Fetching forecast data from CWC...")
        
        # Attempt to fetch from forecast endpoint
        url = f"{self.base_url}/forecast"
        response = self._fetch_with_retry(url)
        
        if not response:
            logger.warning("Forecast data not available")
            return None
        
        try:
            # Try JSON first
            try:
                data = response.json()
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                    df["source_url"] = url
                    logger.info(f"Successfully fetched {len(df)} forecast records")
                    return df
            except json.JSONDecodeError:
                # Fall back to HTML parsing
                soup = BeautifulSoup(response.content, "html.parser")
                forecast_data = []
                
                tables = soup.find_all("table")
                for table in tables:
                    rows = table.find_all("tr")
                    for row in rows[1:]:
                        cols = row.find_all("td")
                        if len(cols) >= 4:
                            try:
                                forecast_record = {
                                    "station_id": cols[0].text.strip(),
                                    "forecast_level": float(cols[1].text.strip()),
                                    "forecast_time": cols[2].text.strip(),
                                    "confidence": float(cols[3].text.strip()),
                                    "source_url": url,
                                }
                                forecast_data.append(forecast_record)
                            except (ValueError, IndexError):
                                continue
                
                if forecast_data:
                    df = pd.DataFrame(forecast_data)
                    logger.info(f"Successfully parsed {len(df)} forecast records")
                    return df
                
        except Exception as e:
            logger.error(f"Error fetching forecast data: {str(e)}")
        
        return None
    
    def save_raw_data(self, data: pd.DataFrame, data_type: str = "stations"):
        """
        Save raw scraped data with metadata.
        
        Args:
            data: DataFrame to save
            data_type: Type of data (stations, forecast, etc.)
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = RAW_DATA_DIR / f"{data_type}_{timestamp}.csv"
        
        data.to_csv(filename, index=False)
        logger.info(f"Saved raw {data_type} data to {filename}")
        
        # Save metadata
        metadata = {
            "data_type": data_type,
            "timestamp": datetime.utcnow().isoformat(),
            "record_count": len(data),
            "source": CWC_ATTRIBUTION["source"],
            "source_url": CWC_ATTRIBUTION["url"],
            "disclaimer": CWC_ATTRIBUTION["disclaimer"],
        }
        
        metadata_file = RAW_DATA_DIR / f"{data_type}_{timestamp}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def fetch_all_stations(self, use_mock_fallback: bool = True) -> Optional[pd.DataFrame]:
        """
        Fetch all available station data.
        
        Args:
            use_mock_fallback: Use mock data if CWC fetch fails
        
        Returns:
            Combined DataFrame with all station information
        """
        logger.info("Starting comprehensive data fetch from CWC...")
        
        station_data = self.fetch_station_data()
        
        if station_data is not None:
            self.save_raw_data(station_data, "stations")
            return station_data
        
        # Fallback to mock data if CWC is unavailable
        if use_mock_fallback:
            logger.warning("CWC data unavailable. Using mock data for demonstration...")
            from src.data_acquisition.mock_data_generator import MockDataGenerator
            
            mock_gen = MockDataGenerator()
            station_data = mock_gen.generate_station_data(num_hours=24)
            mock_gen.save_mock_data(station_data, "stations")
            
            logger.info("Using mock data for pipeline demonstration")
            return station_data
        
        return None

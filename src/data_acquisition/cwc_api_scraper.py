"""
CWC Real-time Hydrograph Data Scraper.

Data Source: Central Water Commission (CWC), Government of India
Website: https://ffs.india-water.gov.in/

Scrapes real-time water level and hydrograph data from CWC's official
Flood Forecasting System (FFS) website.

API Endpoints:
- Station List: https://ffs.india-water.gov.in/#/main/site
- Hydrograph Data: https://ffs.india-water.gov.in/#/main/hydrograph
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

from src.utils.logger import setup_logger
from src.utils.constants import CWC_ATTRIBUTION, RAW_DATA_DIR

logger = setup_logger(__name__)


class CWCAPIScraper:
    """
    Scrapes real-time hydrograph data from CWC's Flood Forecasting System.
    
    Uses the actual CWC API endpoints to fetch:
    - Station information and metadata
    - Real-time water levels
    - Historical hydrograph data
    - Forecast data
    """
    
    # CWC API endpoints
    BASE_URL = "https://ffs.india-water.gov.in"
    API_BASE = "https://ffs.india-water.gov.in/api"
    
    # Endpoints
    STATIONS_ENDPOINT = f"{API_BASE}/stations"
    HYDROGRAPH_ENDPOINT = f"{API_BASE}/hydrograph"
    FORECAST_ENDPOINT = f"{API_BASE}/forecast"
    
    def __init__(self):
        """Initialize CWC API scraper."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("CWC API Scraper initialized")
        logger.info(f"Data source: {CWC_ATTRIBUTION['source']}")
    
    def fetch_stations_from_excel(self, excel_path: str = 'TableViewStationForecastData.xlsx') -> pd.DataFrame:
        """
        Load station data from Excel file.
        
        Args:
            excel_path: Path to the Excel file with station data
            
        Returns:
            DataFrame with station information
        """
        try:
            logger.info(f"Loading station data from {excel_path}...")
            df = pd.read_excel(excel_path)
            logger.info(f"Loaded {len(df)} stations from Excel")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            return pd.DataFrame()
    
    def fetch_station_hydrograph(self, station_name: str, days: int = 7) -> Optional[pd.DataFrame]:
        """
        Fetch hydrograph data for a specific station.
        
        Args:
            station_name: Name of the station
            days: Number of days of historical data
            
        Returns:
            DataFrame with hydrograph data or None
        """
        try:
            logger.info(f"Fetching hydrograph for {station_name}...")
            
            # Try to fetch from CWC API
            params = {
                'station': station_name,
                'days': days
            }
            
            response = self.session.get(
                self.HYDROGRAPH_ENDPOINT,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list):
                    df = pd.DataFrame(data)
                    logger.info(f"Fetched {len(df)} hydrograph records for {station_name}")
                    return df
            else:
                logger.warning(f"API returned status {response.status_code} for {station_name}")
        
        except Exception as e:
            logger.warning(f"Error fetching hydrograph for {station_name}: {str(e)}")
        
        return None
    
    def fetch_station_current_level(self, station_name: str) -> Optional[Dict]:
        """
        Fetch current water level for a station.
        
        Args:
            station_name: Name of the station
            
        Returns:
            Dictionary with current level data or None
        """
        try:
            logger.info(f"Fetching current level for {station_name}...")
            
            params = {'station': station_name}
            
            response = self.session.get(
                f"{self.API_BASE}/current-level",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Fetched current level for {station_name}")
                return data
        
        except Exception as e:
            logger.warning(f"Error fetching current level for {station_name}: {str(e)}")
        
        return None
    
    def fetch_all_stations_data(self, excel_path: str = 'TableViewStationForecastData.xlsx',
                               batch_size: int = 10, delay: float = 0.5) -> pd.DataFrame:
        """
        Fetch current data for all stations from Excel.
        
        Args:
            excel_path: Path to Excel file with station list
            batch_size: Number of stations to fetch before delay
            delay: Delay between requests (seconds)
            
        Returns:
            DataFrame with all station data
        """
        logger.info("Fetching data for all stations...")
        
        # Load station list from Excel
        stations_df = self.fetch_stations_from_excel(excel_path)
        
        if stations_df.empty:
            logger.error("No stations loaded from Excel")
            return pd.DataFrame()
        
        all_data = []
        
        for idx, row in stations_df.iterrows():
            station_name = row['Station Name']
            
            try:
                # Fetch current level
                current_data = self.fetch_station_current_level(station_name)
                
                if current_data:
                    # Combine with station metadata
                    record = {
                        'station_id': f"STN_{idx:04d}",
                        'station_name': station_name,
                        'river_name': row.get('River Name', ''),
                        'basin': row.get('Basin Name', ''),
                        'state': row.get('State name', ''),
                        'district': row.get('District / Town', ''),
                        'latitude': row.get('Latitude', 0),
                        'longitude': row.get('longitude', 0),
                        'division': row.get('Division Name', ''),
                        'site_type': row.get('Type Of Site', ''),
                        'timestamp': datetime.utcnow().isoformat(),
                        'source_url': 'https://ffs.india-water.gov.in/',
                        **current_data  # Merge current level data
                    }
                    all_data.append(record)
                    logger.info(f"[{idx+1}/{len(stations_df)}] Fetched data for {station_name}")
                
                # Rate limiting
                if (idx + 1) % batch_size == 0:
                    logger.info(f"Batch complete. Waiting {delay}s before next batch...")
                    time.sleep(delay)
            
            except Exception as e:
                logger.warning(f"Error processing {station_name}: {str(e)}")
                continue
        
        if all_data:
            df = pd.DataFrame(all_data)
            logger.info(f"Successfully fetched data for {len(df)} stations")
            return df
        else:
            logger.warning("No data fetched for any station")
            return pd.DataFrame()
    
    def fetch_hydrograph_for_all_stations(self, excel_path: str = 'TableViewStationForecastData.xlsx',
                                         days: int = 7, batch_size: int = 5) -> Dict[str, pd.DataFrame]:
        """
        Fetch hydrograph data for all stations.
        
        Args:
            excel_path: Path to Excel file with station list
            days: Number of days of historical data
            batch_size: Number of stations per batch
            
        Returns:
            Dictionary mapping station names to hydrograph DataFrames
        """
        logger.info(f"Fetching hydrograph data for all stations ({days} days)...")
        
        # Load station list
        stations_df = self.fetch_stations_from_excel(excel_path)
        
        if stations_df.empty:
            return {}
        
        hydrographs = {}
        
        for idx, row in stations_df.iterrows():
            station_name = row['Station Name']
            
            try:
                # Fetch hydrograph
                hydrograph_df = self.fetch_station_hydrograph(station_name, days)
                
                if hydrograph_df is not None and not hydrograph_df.empty:
                    hydrographs[station_name] = hydrograph_df
                    logger.info(f"[{idx+1}/{len(stations_df)}] Fetched hydrograph for {station_name}")
                
                # Rate limiting
                if (idx + 1) % batch_size == 0:
                    time.sleep(1)
            
            except Exception as e:
                logger.warning(f"Error fetching hydrograph for {station_name}: {str(e)}")
                continue
        
        logger.info(f"Successfully fetched hydrographs for {len(hydrographs)} stations")
        return hydrographs
    
    def save_station_data(self, data: pd.DataFrame, data_type: str = "stations") -> None:
        """
        Save station data with metadata.
        
        Args:
            data: DataFrame to save
            data_type: Type of data
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = RAW_DATA_DIR / f"{data_type}_{timestamp}.csv"
        
        data.to_csv(filename, index=False)
        logger.info(f"Saved {data_type} data to {filename}")
        
        # Save metadata
        metadata = {
            'data_type': data_type,
            'timestamp': datetime.utcnow().isoformat(),
            'record_count': len(data),
            'source': CWC_ATTRIBUTION['source'],
            'source_url': CWC_ATTRIBUTION['url'],
            'disclaimer': CWC_ATTRIBUTION['disclaimer'],
            'note': 'Real-time data from CWC Flood Forecasting System',
        }
        
        metadata_file = RAW_DATA_DIR / f"{data_type}_{timestamp}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def run_full_scrape(self, excel_path: str = 'TableViewStationForecastData.xlsx') -> Dict:
        """
        Run complete scraping workflow.
        
        Args:
            excel_path: Path to Excel file with station list
            
        Returns:
            Dictionary with scraping results
        """
        logger.info("="*60)
        logger.info("CWC FULL DATA SCRAPE")
        logger.info("="*60)
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'source': CWC_ATTRIBUTION['source'],
        }
        
        try:
            # Fetch current station data
            logger.info("\n1. Fetching current station data...")
            stations_data = self.fetch_all_stations_data(excel_path)
            
            if not stations_data.empty:
                self.save_station_data(stations_data, "stations_current")
                results['stations_count'] = len(stations_data)
                results['stations_data'] = stations_data
            else:
                logger.warning("No current station data fetched")
                results['stations_count'] = 0
            
            # Fetch hydrograph data
            logger.info("\n2. Fetching hydrograph data...")
            hydrographs = self.fetch_hydrograph_for_all_stations(excel_path, days=7)
            
            if hydrographs:
                results['hydrographs_count'] = len(hydrographs)
                results['hydrographs'] = hydrographs
            else:
                logger.warning("No hydrograph data fetched")
                results['hydrographs_count'] = 0
            
            logger.info("\n" + "="*60)
            logger.info("SCRAPING COMPLETE")
            logger.info("="*60)
        
        except Exception as e:
            logger.error(f"Error during scraping: {str(e)}")
            results['error'] = str(e)
        
        return results

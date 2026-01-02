"""
Mock Data Generator for Testing and Demonstration.

Generates realistic flood data for Indian river basins when CWC website is unavailable.
This is for testing and demonstration purposes only.

Data Source: Central Water Commission (CWC), Government of India
Website: https://inf.cwc.gov.in/
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import json
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.constants import RAW_DATA_DIR, CWC_ATTRIBUTION

logger = setup_logger(__name__)


class MockDataGenerator:
    """
    Generates realistic mock flood data for Indian river basins.
    
    Useful for:
    - Testing the pipeline without live CWC data
    - Demonstration purposes
    - Development and debugging
    """
    
    # Indian river basins and stations
    BASINS = {
        'Ganga': {
            'river': 'Ganga',
            'state': 'Uttar Pradesh',
            'stations': [
                {'id': 'G001', 'name': 'Kanpur', 'lat': 26.45, 'lon': 80.23},
                {'id': 'G002', 'name': 'Varanasi', 'lat': 25.32, 'lon': 82.98},
                {'id': 'G003', 'name': 'Patna', 'lat': 25.59, 'lon': 85.13},
            ]
        },
        'Brahmaputra': {
            'river': 'Brahmaputra',
            'state': 'Assam',
            'stations': [
                {'id': 'B001', 'name': 'Guwahati', 'lat': 26.18, 'lon': 91.74},
                {'id': 'B002', 'name': 'Dibrugarh', 'lat': 27.48, 'lon': 94.91},
            ]
        },
        'Godavari': {
            'river': 'Godavari',
            'state': 'Telangana',
            'stations': [
                {'id': 'GO001', 'name': 'Rajahmundry', 'lat': 17.37, 'lon': 81.78},
                {'id': 'GO002', 'name': 'Nashik', 'lat': 19.99, 'lon': 73.79},
            ]
        },
        'Krishna': {
            'river': 'Krishna',
            'state': 'Karnataka',
            'stations': [
                {'id': 'K001', 'name': 'Vijayawada', 'lat': 16.51, 'lon': 80.64},
                {'id': 'K002', 'name': 'Sangli', 'lat': 16.86, 'lon': 75.63},
            ]
        },
        'Yamuna': {
            'river': 'Yamuna',
            'state': 'Delhi',
            'stations': [
                {'id': 'Y001', 'name': 'Delhi', 'lat': 28.61, 'lon': 77.23},
                {'id': 'Y002', 'name': 'Agra', 'lat': 27.18, 'lon': 78.01},
            ]
        }
    }
    
    def __init__(self):
        """Initialize mock data generator."""
        logger.info("Mock Data Generator initialized")
    
    def generate_station_data(self, num_hours: int = 24) -> pd.DataFrame:
        """
        Generate realistic mock station data.
        
        Args:
            num_hours: Number of hours of historical data to generate
            
        Returns:
            DataFrame with mock station data
        """
        logger.info(f"Generating mock station data for {num_hours} hours...")
        
        records = []
        base_time = datetime.utcnow()  # Use naive datetime
        
        for basin_name, basin_data in self.BASINS.items():
            for station in basin_data['stations']:
                # Generate time series for this station
                for hour_offset in range(num_hours):
                    timestamp = base_time - timedelta(hours=hour_offset)
                    
                    # Realistic water level with trend
                    base_level = np.random.uniform(3.0, 6.0)
                    trend_factor = np.sin(hour_offset / 12) * 0.5  # Daily cycle
                    noise = np.random.normal(0, 0.1)
                    current_level = base_level + trend_factor + noise
                    
                    # Thresholds (typical for Indian rivers)
                    warning_level = current_level + np.random.uniform(1.5, 2.5)
                    danger_level = warning_level + np.random.uniform(0.8, 1.5)
                    highest_flood_level = danger_level + np.random.uniform(0.5, 1.0)
                    
                    # Trend
                    if hour_offset > 0:
                        trend = 'rising' if trend_factor > 0 else 'falling'
                    else:
                        trend = 'stable'
                    
                    record = {
                        'station_id': station['id'],
                        'station_name': station['name'],
                        'river_name': basin_data['river'],
                        'basin': basin_name,
                        'state': basin_data['state'],
                        'latitude': station['lat'],
                        'longitude': station['lon'],
                        'current_level': round(current_level, 2),
                        'warning_level': round(warning_level, 2),
                        'danger_level': round(danger_level, 2),
                        'highest_flood_level': round(highest_flood_level, 2),
                        'trend': trend,
                        'timestamp': timestamp.isoformat(),  # Naive ISO format
                        'source_url': 'https://inf.cwc.gov.in/',
                    }
                    records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} mock station records")
        
        return df
    
    def generate_forecast_data(self) -> pd.DataFrame:
        """
        Generate mock forecast data.
        
        Returns:
            DataFrame with mock forecast data
        """
        logger.info("Generating mock forecast data...")
        
        records = []
        base_time = datetime.utcnow()
        
        for basin_name, basin_data in self.BASINS.items():
            for station in basin_data['stations']:
                # Generate forecasts for next 6 hours
                for hour_ahead in range(1, 7):
                    forecast_time = base_time + timedelta(hours=hour_ahead)
                    
                    # Forecast level (with uncertainty)
                    base_level = np.random.uniform(3.0, 6.0)
                    forecast_level = base_level + np.random.uniform(-0.5, 1.0)
                    confidence = np.random.uniform(0.7, 0.95)
                    
                    record = {
                        'station_id': station['id'],
                        'station_name': station['name'],
                        'forecast_level': round(forecast_level, 2),
                        'forecast_time': forecast_time.isoformat(),
                        'confidence': round(confidence, 2),
                        'source_url': 'https://inf.cwc.gov.in/',
                    }
                    records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} mock forecast records")
        
        return df
    
    def save_mock_data(self, data: pd.DataFrame, data_type: str = "stations") -> None:
        """
        Save mock data with metadata.
        
        Args:
            data: DataFrame to save
            data_type: Type of data (stations, forecast, etc.)
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = RAW_DATA_DIR / f"{data_type}_{timestamp}.csv"
        
        data.to_csv(filename, index=False)
        logger.info(f"Saved mock {data_type} data to {filename}")
        
        # Save metadata
        metadata = {
            'data_type': data_type,
            'timestamp': datetime.utcnow().isoformat(),
            'record_count': len(data),
            'source': CWC_ATTRIBUTION['source'],
            'source_url': CWC_ATTRIBUTION['url'],
            'disclaimer': CWC_ATTRIBUTION['disclaimer'],
            'note': 'This is mock/synthetic data for testing and demonstration purposes only.',
        }
        
        metadata_file = RAW_DATA_DIR / f"{data_type}_{timestamp}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def generate_all_mock_data(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all mock data.
        
        Returns:
            Dictionary with station and forecast data
        """
        logger.info("Generating all mock data...")
        
        station_data = self.generate_station_data(num_hours=24)
        forecast_data = self.generate_forecast_data()
        
        # Save data
        self.save_mock_data(station_data, "stations")
        self.save_mock_data(forecast_data, "forecast")
        
        return {
            'stations': station_data,
            'forecast': forecast_data
        }
    
    def get_sample_data(self) -> pd.DataFrame:
        """Get a small sample of mock data for quick testing."""
        logger.info("Generating sample mock data...")
        
        records = []
        base_time = datetime.utcnow()
        
        # Just a few stations for quick testing
        sample_stations = [
            {'id': 'G001', 'name': 'Kanpur', 'river': 'Ganga', 'basin': 'Ganga', 'state': 'UP'},
            {'id': 'B001', 'name': 'Guwahati', 'river': 'Brahmaputra', 'basin': 'Brahmaputra', 'state': 'Assam'},
            {'id': 'GO001', 'name': 'Rajahmundry', 'river': 'Godavari', 'basin': 'Godavari', 'state': 'Telangana'},
        ]
        
        for station in sample_stations:
            current_level = np.random.uniform(3.0, 6.0)
            
            record = {
                'station_id': station['id'],
                'station_name': station['name'],
                'river_name': station['river'],
                'basin': station['basin'],
                'state': station['state'],
                'current_level': round(current_level, 2),
                'warning_level': round(current_level + 2.0, 2),
                'danger_level': round(current_level + 3.0, 2),
                'highest_flood_level': round(current_level + 4.0, 2),
                'trend': np.random.choice(['rising', 'falling', 'stable']),
                'timestamp': base_time.isoformat(),
                'source_url': 'https://inf.cwc.gov.in/',
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} sample records")
        
        return df

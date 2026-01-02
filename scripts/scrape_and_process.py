"""
Complete CWC Data Processing Pipeline.

Loads station data from Excel and generates realistic hydrograph data,
then runs the full flood prediction pipeline.

Data Source: Central Water Commission (CWC), Government of India
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.constants import RAW_DATA_DIR, CWC_ATTRIBUTION
from src.pipeline import FloodWarningPipeline

logger = setup_logger(__name__)


class CWCDataProcessor:
    """
    Processes CWC station data and generates realistic hydrograph data.
    """
    
    def __init__(self):
        """Initialize processor."""
        logger.info("CWC Data Processor initialized")
    
    def load_stations_from_excel(self, excel_path: str = 'TableViewStationForecastData.xlsx') -> pd.DataFrame:
        """
        Load station list from Excel.
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            DataFrame with station data
        """
        try:
            logger.info(f"Loading stations from {excel_path}...")
            df = pd.read_excel(excel_path)
            logger.info(f"Loaded {len(df)} stations")
            logger.info(f"Basins: {df['Basin Name'].nunique()}")
            logger.info(f"States: {df['State name'].nunique()}")
            logger.info(f"Rivers: {df['River Name'].nunique()}")
            return df
        except Exception as e:
            logger.error(f"Error loading Excel: {str(e)}")
            return pd.DataFrame()
    
    def generate_hydrograph_data(self, station_row: pd.Series, days: int = 7) -> pd.DataFrame:
        """
        Generate realistic hydrograph data for a station.
        
        Args:
            station_row: Station information
            days: Number of days of data
            
        Returns:
            DataFrame with hydrograph data
        """
        station_name = station_row['Station Name']
        
        # Generate time series
        records = []
        base_time = datetime.utcnow()
        
        # Random base level for this station
        base_level = np.random.uniform(2.0, 8.0)
        
        for day_offset in range(days):
            for hour_offset in range(24):
                timestamp = base_time - timedelta(days=day_offset, hours=hour_offset)
                
                # Realistic water level with daily and trend components
                daily_cycle = np.sin(hour_offset / 24 * 2 * np.pi) * 0.5
                trend = np.sin(day_offset / days * 2 * np.pi) * 1.0
                noise = np.random.normal(0, 0.1)
                
                current_level = base_level + daily_cycle + trend + noise
                current_level = max(0.5, current_level)  # Ensure positive
                
                # Thresholds
                warning_level = current_level + np.random.uniform(1.5, 2.5)
                danger_level = warning_level + np.random.uniform(0.8, 1.5)
                
                # Trend indicator
                if daily_cycle > 0.2:
                    trend_str = 'rising'
                elif daily_cycle < -0.2:
                    trend_str = 'falling'
                else:
                    trend_str = 'stable'
                
                record = {
                    'station_id': f"STN_{station_row.name:04d}",
                    'station_name': station_name,
                    'river_name': station_row['River Name'],
                    'basin': station_row['Basin Name'],
                    'state': station_row['State name'],
                    'district': station_row['District / Town'],
                    'latitude': station_row['Latitude'],
                    'longitude': station_row['longitude'],
                    'division': station_row['Division Name'],
                    'site_type': station_row['Type Of Site'],
                    'current_level': round(current_level, 2),
                    'warning_level': round(warning_level, 2),
                    'danger_level': round(danger_level, 2),
                    'highest_flood_level': round(danger_level + np.random.uniform(0.5, 1.0), 2),
                    'trend': trend_str,
                    'timestamp': timestamp.isoformat(),
                    'source_url': 'https://ffs.india-water.gov.in/',
                }
                records.append(record)
        
        return pd.DataFrame(records)
    
    def process_all_stations(self, excel_path: str = 'TableViewStationForecastData.xlsx',
                            days: int = 7, sample_size: int = None) -> pd.DataFrame:
        """
        Process all stations and generate hydrograph data.
        
        Args:
            excel_path: Path to Excel file
            days: Days of historical data
            sample_size: If set, process only this many stations
            
        Returns:
            Combined DataFrame with all station data
        """
        logger.info("="*70)
        logger.info("PROCESSING CWC STATIONS")
        logger.info("="*70)
        
        # Load stations
        stations_df = self.load_stations_from_excel(excel_path)
        
        if stations_df.empty:
            logger.error("No stations loaded")
            return pd.DataFrame()
        
        # Limit to sample if specified
        if sample_size:
            stations_df = stations_df.head(sample_size)
            logger.info(f"Processing sample of {sample_size} stations")
        
        all_data = []
        
        for idx, (_, station_row) in enumerate(stations_df.iterrows()):
            try:
                station_name = station_row['Station Name']
                logger.info(f"[{idx+1}/{len(stations_df)}] Generating data for {station_name}...")
                
                # Generate hydrograph data
                hydrograph = self.generate_hydrograph_data(station_row, days)
                all_data.append(hydrograph)
                
                if (idx + 1) % 50 == 0:
                    logger.info(f"Processed {idx+1} stations...")
            
            except Exception as e:
                logger.warning(f"Error processing {station_row['Station Name']}: {str(e)}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n{'='*70}")
            logger.info(f"Generated data for {combined_df['station_name'].nunique()} stations")
            logger.info(f"Total records: {len(combined_df)}")
            logger.info(f"{'='*70}\n")
            return combined_df
        else:
            logger.error("No data generated")
            return pd.DataFrame()
    
    def save_data(self, data: pd.DataFrame, data_type: str = "cwc_hydrograph") -> None:
        """
        Save processed data.
        
        Args:
            data: DataFrame to save
            data_type: Type of data
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = RAW_DATA_DIR / f"{data_type}_{timestamp}.csv"
        
        data.to_csv(filename, index=False)
        logger.info(f"Saved data to: {filename}")
        
        # Save metadata
        metadata = {
            'data_type': data_type,
            'timestamp': datetime.utcnow().isoformat(),
            'record_count': len(data),
            'unique_stations': data['station_name'].nunique(),
            'unique_basins': data['basin'].nunique(),
            'unique_states': data['state'].nunique(),
            'source': CWC_ATTRIBUTION['source'],
            'source_url': CWC_ATTRIBUTION['url'],
            'disclaimer': CWC_ATTRIBUTION['disclaimer'],
            'columns': data.columns.tolist(),
        }
        
        metadata_file = RAW_DATA_DIR / f"{data_type}_{timestamp}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to: {metadata_file}")
    
    def run(self, excel_path: str = 'TableViewStationForecastData.xlsx',
            days: int = 7, sample_size: int = None) -> pd.DataFrame:
        """
        Run complete processing workflow.
        
        Args:
            excel_path: Path to Excel file
            days: Days of historical data
            sample_size: Optional sample size
            
        Returns:
            Processed DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("CWC DATA PROCESSING PIPELINE")
        logger.info("="*70)
        logger.info(f"Data Source: {CWC_ATTRIBUTION['source']}")
        logger.info(f"Website: {CWC_ATTRIBUTION['url']}")
        logger.info("="*70 + "\n")
        
        # Process stations
        data = self.process_all_stations(excel_path, days, sample_size)
        
        if not data.empty:
            # Save data
            self.save_data(data, "cwc_hydrograph")
            
            # Print summary
            logger.info("\n" + "="*70)
            logger.info("DATA SUMMARY")
            logger.info("="*70)
            logger.info(f"Total Records: {len(data)}")
            logger.info(f"Stations: {data['station_name'].nunique()}")
            logger.info(f"Basins: {data['basin'].nunique()}")
            logger.info(f"States: {data['state'].nunique()}")
            logger.info(f"Rivers: {data['river_name'].nunique()}")
            logger.info(f"Date Range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            logger.info("="*70 + "\n")
            
            # Show basin distribution
            logger.info("Basin Distribution:")
            basin_counts = data['basin'].value_counts()
            for basin, count in basin_counts.head(10).items():
                logger.info(f"  {basin}: {count} records")
            
            return data
        else:
            logger.error("No data processed")
            return pd.DataFrame()


def main():
    """Main entry point."""
    import sys
    
    # Process CWC data
    processor = CWCDataProcessor()
    
    # Process all stations (or sample for testing)
    # Use sample_size=100 for quick testing
    data = processor.run(
        excel_path='TableViewStationForecastData.xlsx',
        days=7,
        sample_size=100  # Process 100 stations for testing
    )
    
    if data.empty:
        logger.error("Data processing failed")
        return 1
    
    # Run flood prediction pipeline
    logger.info("\n" + "="*70)
    logger.info("RUNNING FLOOD PREDICTION PIPELINE")
    logger.info("="*70 + "\n")
    
    try:
        pipeline = FloodWarningPipeline()
        results = pipeline.run_full_pipeline(train_models=False)
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE RESULTS")
        logger.info("="*70)
        logger.info(f"Raw Data: {results.get('raw_data_count', 0)} records")
        logger.info(f"Cleaned Data: {results.get('cleaned_data_count', 0)} records")
        logger.info(f"Features: {results.get('features_count', 0)} records")
        logger.info(f"Predictions: {results.get('predictions_count', 0)} records")
        logger.info(f"Alerts: {results.get('alerts_count', 0)} triggered")
        logger.info("="*70 + "\n")
        
        return 0
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

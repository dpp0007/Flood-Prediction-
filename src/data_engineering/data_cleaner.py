"""
Data Cleaning and Validation Module.

Data Source: Central Water Commission (CWC), Government of India
Website: https://inf.cwc.gov.in/

Handles:
- Missing value imputation
- Unit conversion
- Sensor error detection
- Data validation
- Outlier flagging
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime

from src.utils.logger import setup_logger
from src.utils.constants import (
    MAX_WATER_LEVEL_JUMP_METERS,
    MIN_VALID_LEVEL,
    MAX_VALID_LEVEL,
    CWC_ATTRIBUTION,
)

logger = setup_logger(__name__)


class DataCleaner:
    """
    Cleans and validates hydrological data from CWC.
    """
    
    def __init__(self):
        """Initialize data cleaner."""
        logger.info("DataCleaner initialized")
        self.validation_report = {}
    
    def validate_water_levels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate water level measurements.
        
        Args:
            df: DataFrame with water level data
            
        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        logger.info(f"Validating {len(df)} records...")
        
        report = {
            "total_records": len(df),
            "invalid_levels": 0,
            "missing_values": 0,
            "sensor_errors": 0,
            "flagged_records": 0,
        }
        
        df = df.copy()
        df["is_valid"] = True
        df["validation_flags"] = ""
        
        # Check for missing values
        missing_mask = df["current_level"].isna()
        report["missing_values"] = missing_mask.sum()
        df.loc[missing_mask, "is_valid"] = False
        df.loc[missing_mask, "validation_flags"] += "MISSING_VALUE;"
        
        # Check for out-of-range values
        out_of_range = (df["current_level"] < MIN_VALID_LEVEL) | (df["current_level"] > MAX_VALID_LEVEL)
        report["invalid_levels"] = out_of_range.sum()
        df.loc[out_of_range, "is_valid"] = False
        df.loc[out_of_range, "validation_flags"] += "OUT_OF_RANGE;"
        
        # Detect sensor errors (sudden jumps)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values(["station_id", "timestamp"])
            
            for station_id in df["station_id"].unique():
                station_data = df[df["station_id"] == station_id].copy()
                if len(station_data) > 1:
                    level_diff = station_data["current_level"].diff().abs()
                    sensor_error_mask = level_diff > MAX_WATER_LEVEL_JUMP_METERS
                    
                    if sensor_error_mask.any():
                        report["sensor_errors"] += sensor_error_mask.sum()
                        df.loc[station_data.index[sensor_error_mask.values], "is_valid"] = False
                        df.loc[station_data.index[sensor_error_mask.values], "validation_flags"] += "SENSOR_ERROR;"
        
        # Flag records with validation issues
        report["flagged_records"] = (~df["is_valid"]).sum()
        
        logger.info(f"Validation report: {report}")
        self.validation_report = report
        
        return df, report
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = "interpolate",
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in data.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'interpolate', 'forward_fill', or 'drop'
            columns: Columns to apply strategy to
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        if columns is None:
            columns = ["current_level", "warning_level", "danger_level"]
        
        for col in columns:
            if col not in df.columns:
                continue
            
            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue
            
            logger.info(f"Handling {missing_count} missing values in {col} using {strategy}")
            
            if strategy == "interpolate":
                df[col] = df.groupby("station_id")[col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both")
                )
            elif strategy == "forward_fill":
                df[col] = df.groupby("station_id")[col].transform(lambda x: x.fillna(method="ffill"))
            elif strategy == "drop":
                df = df.dropna(subset=[col])
            
            remaining_missing = df[col].isna().sum()
            if remaining_missing > 0:
                logger.warning(f"Still {remaining_missing} missing values in {col} after {strategy}")
        
        return df
    
    def normalize_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize measurement units.
        
        Args:
            df: DataFrame with measurements
            
        Returns:
            DataFrame with normalized units
        """
        df = df.copy()
        
        # Ensure water levels are in meters
        level_columns = ["current_level", "warning_level", "danger_level", "highest_flood_level"]
        for col in level_columns:
            if col in df.columns:
                # Check if values seem to be in cm (> 1000) and convert to meters
                if (df[col] > 1000).any():
                    logger.info(f"Converting {col} from cm to meters")
                    df[col] = df[col] / 100
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, column: str = "current_level") -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        
        Args:
            df: DataFrame with data
            column: Column to check for outliers
            
        Returns:
            DataFrame with outlier flag
        """
        df = df.copy()
        
        if column not in df.columns:
            return df
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        df["is_outlier"] = outlier_mask
        
        outlier_count = outlier_mask.sum()
        logger.info(f"Detected {outlier_count} outliers in {column}")
        
        return df
    
    def clean(
        self,
        df: pd.DataFrame,
        validate: bool = True,
        handle_missing: bool = True,
        normalize: bool = True,
        detect_outliers: bool = True
    ) -> pd.DataFrame:
        """
        Complete data cleaning pipeline.
        
        Args:
            df: Raw DataFrame
            validate: Whether to validate data
            handle_missing: Whether to handle missing values
            normalize: Whether to normalize units
            detect_outliers: Whether to detect outliers
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        
        df = df.copy()
        
        if validate:
            df, _ = self.validate_water_levels(df)
        
        if normalize:
            df = self.normalize_units(df)
        
        if handle_missing:
            df = self.handle_missing_values(df)
        
        if detect_outliers:
            df = self.detect_outliers(df)
        
        logger.info(f"Data cleaning complete. Final shape: {df.shape}")
        
        return df
    
    def get_validation_report(self) -> Dict:
        """Get the last validation report."""
        return self.validation_report

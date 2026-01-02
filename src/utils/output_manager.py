"""
Standardized Output Manager for Flood Prediction System.

Handles:
- JSON prediction output generation
- Explicit data/artifact saving at each stage
- Directory structure management
- Runtime visibility and logging
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OutputManager:
    """
    Manages all output generation and artifact storage.
    Ensures explicit saving and logging at every stage.
    """
    
    def __init__(self, base_dir: str = "data"):
        """
        Initialize output manager.
        
        Args:
            base_dir: Base directory for all outputs
        """
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
        
        # Define all output directories
        self.dirs = {
            'raw': self.base_dir / 'raw',
            'processed': self.base_dir / 'processed',
            'features': self.base_dir / 'features',
            'models': self.base_dir / 'models',
            'outputs_tabular': self.base_dir / 'outputs' / 'tabular',
            'outputs_json': self.base_dir / 'outputs' / 'json_predictions',
            'outputs_plots': self.base_dir / 'outputs' / 'plots',
        }
        
        # Create all directories
        self._create_directories()
        
        logger.info("OutputManager initialized")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Timestamp: {self.timestamp}")
    
    def _create_directories(self) -> None:
        """Create all required directories."""
        for dir_name, dir_path in self.dirs.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ready: {dir_path}")
    
    def save_raw_data(self, data: pd.DataFrame, source_url: str = "",
                     scraper_version: str = "1.0") -> str:
        """
        Save raw scraped data with metadata.
        
        Args:
            data: Raw DataFrame from scraper
            source_url: URL of data source
            scraper_version: Version of scraper used
            
        Returns:
            Path to saved file
        """
        filename = f"cwc_raw_{self.timestamp}.csv"
        filepath = self.dirs['raw'] / filename
        
        # Save CSV
        data.to_csv(filepath, index=False)
        logger.info(f"[RAW DATA SAVED] {filepath}")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.utcnow().isoformat(),
            'source_url': source_url,
            'scrape_time': self.timestamp,
            'number_of_records': len(data),
            'scraper_version': scraper_version,
            'columns': data.columns.tolist(),
            'data_types': {col: str(dtype) for col, dtype in data.dtypes.items()},
        }
        
        meta_filename = f"cwc_raw_{self.timestamp}_meta.json"
        meta_filepath = self.dirs['raw'] / meta_filename
        
        with open(meta_filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"[RAW DATA METADATA SAVED] {meta_filepath}")
        logger.info(f"  Records: {len(data)}")
        logger.info(f"  Columns: {len(data.columns)}")
        
        return str(filepath)
    
    def save_cleaned_data(self, data: pd.DataFrame) -> str:
        """
        Save cleaned and validated data.
        
        Args:
            data: Cleaned DataFrame
            
        Returns:
            Path to saved file
        """
        filename = f"cwc_cleaned_{self.timestamp}.csv"
        filepath = self.dirs['processed'] / filename
        
        data.to_csv(filepath, index=False)
        logger.info(f"[CLEANED DATA SAVED] {filepath}")
        
        # Log validation statistics
        if 'is_valid' in data.columns:
            valid_count = data['is_valid'].sum()
            invalid_count = (~data['is_valid']).sum()
            logger.info(f"  Valid records: {valid_count}")
            logger.info(f"  Invalid records: {invalid_count}")
        
        if 'is_outlier' in data.columns:
            outlier_count = data['is_outlier'].sum()
            logger.info(f"  Outliers detected: {outlier_count}")
        
        return str(filepath)
    
    def save_features(self, features: pd.DataFrame) -> str:
        """
        Save engineered features.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Path to saved file
        """
        filename = f"features_{self.timestamp}.csv"
        filepath = self.dirs['features'] / filename
        
        features.to_csv(filepath, index=False)
        logger.info(f"[FEATURES SAVED] {filepath}")
        logger.info(f"  Records: {len(features)}")
        logger.info(f"  Features: {len(features.columns)}")
        logger.info(f"  Feature columns: {list(features.columns)}")
        
        return str(filepath)
    
    def save_predictions_table(self, predictions: pd.DataFrame) -> str:
        """
        Save predictions as tabular data before JSON conversion.
        
        Args:
            predictions: Predictions DataFrame
            
        Returns:
            Path to saved file
        """
        filename = f"predictions_table_{self.timestamp}.csv"
        filepath = self.dirs['outputs_tabular'] / filename
        
        predictions.to_csv(filepath, index=False)
        logger.info(f"[PREDICTIONS TABLE SAVED] {filepath}")
        logger.info(f"  Records: {len(predictions)}")
        logger.info(f"  Columns: {list(predictions.columns)}")
        
        return str(filepath)
    
    def save_prediction_json(self, station_id: str, prediction_data: Dict[str, Any]) -> str:
        """
        Save individual prediction as standardized JSON.
        
        Args:
            station_id: Station identifier
            prediction_data: Prediction dictionary (must match schema)
            
        Returns:
            Path to saved file
        """
        filename = f"predictions_{station_id}_{self.timestamp}.json"
        filepath = self.dirs['outputs_json'] / filename
        
        # Validate schema
        self._validate_prediction_schema(prediction_data)
        
        # Save JSON with 2-space indentation (CHANGE 4)
        with open(filepath, 'w') as f:
            json.dump(prediction_data, f, indent=2, default=str)
        
        logger.debug(f"[JSON PREDICTION SAVED] {filepath}")
        
        return str(filepath)
    
    def cleanup_old_json_files(self) -> int:
        """
        Delete all existing JSON files before generating new ones.
        
        Returns:
            Number of files deleted
        """
        json_dir = self.dirs['outputs_json']
        
        if not json_dir.exists():
            logger.debug(f"JSON directory does not exist yet: {json_dir}")
            return 0
        
        json_files = list(json_dir.glob('predictions_*.json'))
        
        if len(json_files) == 0:
            logger.debug("No old JSON files to clean up")
            return 0
        
        for json_file in json_files:
            try:
                json_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {json_file}: {str(e)}")
        
        logger.info(f"[CLEANUP] Deleted {len(json_files)} old JSON files from {json_dir}")
        
        return len(json_files)
    
    def save_all_predictions_json(self, predictions: pd.DataFrame,
                                  station_metadata: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        Convert predictions table to individual JSON files.
        Filters to only MEDIUM and HIGH risk levels.
        
        Args:
            predictions: Predictions DataFrame
            station_metadata: Optional station metadata for context
            
        Returns:
            Dictionary with statistics about JSON generation
        """
        # Clean up old JSON files first
        self.cleanup_old_json_files()
        
        saved_files = []
        skipped_files = []
        medium_count = 0
        high_count = 0
        
        logger.info(f"[JSON CONVERSION START] Converting {len(predictions)} predictions to JSON")
        logger.info(f"[FILTERING] Generating JSON for MEDIUM, HIGH, and top 3 LOW risk stations")
        
        # Separate predictions by risk level
        medium_predictions = predictions[
            predictions['risk_tier_name'].str.lower() == 'medium'
        ]
        high_predictions = predictions[
            predictions['risk_tier_name'].str.lower() == 'high'
        ]
        low_predictions = predictions[
            predictions['risk_tier_name'].str.lower() == 'low'
        ]
        
        medium_count = len(medium_predictions)
        high_count = len(high_predictions)
        low_count = len(low_predictions)
        
        # Get top 3 LOW risk stations by lowest risk_score
        if len(low_predictions) > 0:
            low_predictions_sorted = low_predictions.sort_values('risk_score')
            top_3_low = low_predictions_sorted.head(3)
            low_json_count = len(top_3_low)
        else:
            top_3_low = pd.DataFrame()
            low_json_count = 0
        
        # Combine predictions to generate JSON for
        predictions_to_json = pd.concat([
            medium_predictions,
            high_predictions,
            top_3_low
        ], ignore_index=True)
        
        logger.info(f"[MEDIUM RISK] {medium_count} stations")
        logger.info(f"[HIGH RISK] {high_count} stations")
        logger.info(f"[LOW RISK] {low_count} stations (generating JSON for top {low_json_count})")
        
        saved_files = []
        
        # Generate JSON for selected predictions
        for idx, row in predictions_to_json.iterrows():
            try:
                station_id = row.get('station_id', f'STATION_{idx}')
                
                # Resolve district correctly
                location_data = self._resolve_location_data(
                    station_id=station_id,
                    row=row,
                    station_metadata=station_metadata
                )
                
                # Build prediction JSON
                prediction_json = self._build_prediction_json(
                    row=row,
                    location_data=location_data,
                    station_id=station_id
                )
                
                # Save JSON
                filepath = self.save_prediction_json(station_id, prediction_json)
                saved_files.append(filepath)
            
            except Exception as e:
                logger.error(f"Error converting prediction for {station_id}: {str(e)}")
                continue
        
        logger.info(f"[JSON CONVERSION COMPLETE] {len(saved_files)} JSON files saved")
        logger.info(f"  Location: {self.dirs['outputs_json']}")
        
        return {
            'total_processed': len(predictions),
            'json_generated': len(saved_files),
            'medium_risk_count': medium_count,
            'high_risk_count': high_count,
            'low_risk_count': low_count,
            'low_risk_json_generated': low_json_count,
            'saved_files': saved_files,
        }
    
    def _resolve_location_data(self, station_id: str, row: pd.Series,
                               station_metadata: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """
        CHANGE 2: Resolve district and location data correctly.
        Also ensures latitude and longitude are populated.
        
        Args:
            station_id: Station identifier
            row: Prediction row
            station_metadata: Station metadata DataFrame
            
        Returns:
            Dictionary with location data (district, state, lat, lon)
        """
        location_data = {
            'district': 'Not Available',
            'state': 'Unknown',
            'latitude': 0.0,
            'longitude': 0.0,
        }
        
        # Try to get from station metadata first
        if station_metadata is not None and not station_metadata.empty:
            station_meta = station_metadata[
                station_metadata['station_id'] == station_id
            ]
            
            if not station_meta.empty:
                station_meta = station_meta.iloc[0]
                
                # Get state
                state = str(station_meta.get('state', 'Unknown')).strip()
                if state and state.lower() != 'unknown':
                    location_data['state'] = state
                
                # Get latitude - try multiple column names
                lat = None
                for lat_col in ['latitude', 'Latitude', 'lat', 'Lat']:
                    if lat_col in station_meta.index:
                        try:
                            lat_val = station_meta.get(lat_col)
                            if lat_val is not None and lat_val != '' and str(lat_val).lower() != 'nan':
                                lat = float(lat_val)
                                if lat != 0.0:
                                    location_data['latitude'] = lat
                                    logger.debug(f"[LAT] {station_id}: Found latitude {lat} from column {lat_col}")
                                    break
                        except (ValueError, TypeError) as e:
                            logger.debug(f"[LAT ERROR] {station_id}: Could not parse {lat_col}={station_meta.get(lat_col)}: {e}")
                            pass
                
                # Get longitude - try multiple column names
                lon = None
                for lon_col in ['longitude', 'Longitude', 'lon', 'Lon', 'lng', 'Lng']:
                    if lon_col in station_meta.index:
                        try:
                            lon_val = station_meta.get(lon_col)
                            if lon_val is not None and lon_val != '' and str(lon_val).lower() != 'nan':
                                lon = float(lon_val)
                                if lon != 0.0:
                                    location_data['longitude'] = lon
                                    logger.debug(f"[LON] {station_id}: Found longitude {lon} from column {lon_col}")
                                    break
                        except (ValueError, TypeError) as e:
                            logger.debug(f"[LON ERROR] {station_id}: Could not parse {lon_col}={station_meta.get(lon_col)}: {e}")
                            pass
                
                # Try to get district
                district = str(station_meta.get('district', '')).strip()
                if district and district.lower() not in ['unknown', 'nan', '']:
                    location_data['district'] = district
                    # Don't return early - continue to ensure lat/lon are populated
        
        # Fallback 1: Try to parse from station_name (only if district not found)
        if location_data['district'] == 'Not Available' and 'station_name' in row:
            station_name = str(row.get('station_name', '')).strip()
            if station_name and station_name.lower() != 'unknown':
                # Try to extract district from station name
                # Format is often: "District, State" or "City, District, State"
                parts = [p.strip() for p in station_name.split(',')]
                if len(parts) >= 2:
                    location_data['district'] = parts[0]
                    logger.debug(f"[DISTRICT FALLBACK 1] {station_id}: Extracted from station_name: {parts[0]}")
        
        # Fallback 2: Use state + river name (only if district still not found)
        if location_data['district'] == 'Not Available' and station_metadata is not None and not station_metadata.empty:
            station_meta = station_metadata[
                station_metadata['station_id'] == station_id
            ]
            
            if not station_meta.empty:
                station_meta = station_meta.iloc[0]
                river_name = str(station_meta.get('river_name', '')).strip()
                state = str(station_meta.get('state', '')).strip()
                
                if river_name and state:
                    location_data['district'] = f"{river_name} Basin, {state}"
                    logger.debug(f"[DISTRICT FALLBACK 2] {station_id}: Using river+state: {location_data['district']}")
        
        # If all fallbacks fail, log warning
        if location_data['district'] == 'Not Available':
            logger.warning(f"[WARNING] District not found for station_id={station_id}, using fallback value: {location_data['district']}")
        
        return location_data
    
    def _build_prediction_json(self, row: pd.Series, location_data: Dict,
                               station_id: str) -> Dict[str, Any]:
        """
        Build standardized prediction JSON from row data.
        
        Args:
            row: Prediction row
            location_data: Location context
            station_id: Station identifier
            
        Returns:
            Standardized prediction dictionary
        """
        risk_score = float(row.get('risk_score', 0.0))
        warning_prob = float(row.get('warning_probability', 0.0))
        risk_tier = str(row.get('risk_tier_name', 'unknown')).lower()
        
        # Determine flood probability (use warning probability)
        flood_probability = warning_prob
        
        # Determine risk level
        if risk_tier == 'high':
            risk_level = 'HIGH'
        elif risk_tier == 'medium':
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Model confidence (average of risk score and warning probability)
        model_confidence = (risk_score + warning_prob) / 2.0
        
        # Generate explanations
        explanations = self._generate_explanations(row, risk_level)
        
        # Check for demo mode metadata
        demo_mode = row.get('demo_mode', False)
        alert_mode = row.get('alert_mode', 'natural')
        
        # Add demo explanation if forced alert
        if demo_mode and alert_mode in ['forced_high', 'forced_medium']:
            explanations.append("Risk level elevated artificially for demonstration/testing purposes.")
        
        # Prediction time and validity
        prediction_time = datetime.utcnow()
        valid_until = prediction_time + timedelta(hours=6)
        
        # Build JSON structure
        prediction_json = {
            'prediction': {
                'flood_probability': round(flood_probability, 4),
                'risk_level': risk_level,
            },
            'confidence': {
                'model_confidence': round(model_confidence, 4),
            },
            'explanation': explanations,
            'context': {
                'location': location_data or {
                    'district': 'Unknown',
                    'state': 'Unknown',
                    'latitude': 0.0,
                    'longitude': 0.0,
                },
                'time_window': {
                    'prediction_time': prediction_time.isoformat() + 'Z',
                    'valid_until': valid_until.isoformat() + 'Z',
                }
            },
            'metadata': {
                'model_type': 'ml_hybrid',
                'model_version': '1.0',
                'data_source': ['Central Water Commission (CWC), Government of India'],
                'disclaimer': 'For research and demonstration purposes only. Does NOT replace official CWC flood warnings.',
                'demo_mode': demo_mode,
                'alert_mode': alert_mode,
            }
        }
        
        return prediction_json
    
    def _generate_explanations(self, row: pd.Series, risk_level: str) -> List[str]:
        """
        Generate human-readable explanations.
        
        Args:
            row: Prediction row
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            
        Returns:
            List of explanation strings
        """
        explanations = []
        
        risk_score = float(row.get('risk_score', 0.0))
        warning_prob = float(row.get('warning_probability', 0.0))
        distance_to_danger = float(row.get('distance_to_danger', 0.0))
        rate_of_rise = float(row.get('rate_of_rise_3h', 0.0))
        
        # Explanation 1: Risk level summary
        if risk_level == 'HIGH':
            explanations.append(
                f"HIGH FLOOD RISK: Risk score {risk_score:.1%} with {warning_prob:.1%} warning probability."
            )
        elif risk_level == 'MEDIUM':
            explanations.append(
                f"MODERATE FLOOD RISK: Risk score {risk_score:.1%} with {warning_prob:.1%} warning probability."
            )
        else:
            explanations.append(
                f"LOW FLOOD RISK: Risk score {risk_score:.1%} with {warning_prob:.1%} warning probability."
            )
        
        # Explanation 2: Distance to danger
        if distance_to_danger < 0.5:
            explanations.append(
                f"CRITICAL: Water level is only {distance_to_danger:.2f}m below danger threshold."
            )
        elif distance_to_danger < 1.0:
            explanations.append(
                f"WARNING: Water level is {distance_to_danger:.2f}m below danger threshold."
            )
        else:
            explanations.append(
                f"Water level is {distance_to_danger:.2f}m below danger threshold."
            )
        
        # Explanation 3: Rate of rise
        if rate_of_rise > 0.15:
            explanations.append(
                f"RAPID RISE: Water level rising at {rate_of_rise:.3f}m/hour."
            )
        elif rate_of_rise > 0.05:
            explanations.append(
                f"Moderate rise: Water level rising at {rate_of_rise:.3f}m/hour."
            )
        else:
            explanations.append(
                f"Water level change: {rate_of_rise:.3f}m/hour."
            )
        
        return explanations
    
    def _validate_prediction_schema(self, data: Dict[str, Any]) -> None:
        """
        Validate prediction JSON schema.
        
        Args:
            data: Prediction dictionary
            
        Raises:
            ValueError: If schema is invalid
        """
        required_keys = ['prediction', 'confidence', 'explanation', 'context', 'metadata']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate prediction
        if 'flood_probability' not in data['prediction']:
            raise ValueError("Missing prediction.flood_probability")
        if 'risk_level' not in data['prediction']:
            raise ValueError("Missing prediction.risk_level")
        
        # Validate confidence
        if 'model_confidence' not in data['confidence']:
            raise ValueError("Missing confidence.model_confidence")
        
        # Validate explanation
        if not isinstance(data['explanation'], list):
            raise ValueError("explanation must be a list")
        
        # Validate context
        if 'location' not in data['context']:
            raise ValueError("Missing context.location")
        if 'time_window' not in data['context']:
            raise ValueError("Missing context.time_window")
    
    def save_plot(self, fig, plot_name: str) -> str:
        """
        Save matplotlib figure.
        
        Args:
            fig: Matplotlib figure object
            plot_name: Name of plot (without extension)
            
        Returns:
            Path to saved file
        """
        filename = f"{plot_name}.png"
        filepath = self.dirs['outputs_plots'] / filename
        
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"[PLOT SAVED] {filepath}")
        
        return str(filepath)
    
    def print_pipeline_summary(self, stage: str, details: Dict[str, Any]) -> None:
        """
        Print clear pipeline stage summary.
        
        Args:
            stage: Pipeline stage name
            details: Stage details dictionary
        """
        print("\n" + "="*60)
        print(f"{stage.upper()} COMPLETE")
        print("="*60)
        
        for key, value in details.items():
            if isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
        
        print("="*60 + "\n")
    
    def validate_outputs(self) -> Dict[str, Any]:
        """
        Validate that all expected outputs exist.
        
        Returns:
            Validation report dictionary
        """
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'directories': {},
            'files': {},
            'validation_passed': True,
        }
        
        # Check directories
        for dir_name, dir_path in self.dirs.items():
            exists = dir_path.exists()
            report['directories'][dir_name] = {
                'path': str(dir_path),
                'exists': exists,
            }
            if not exists:
                report['validation_passed'] = False
        
        # Check files
        for dir_name, dir_path in self.dirs.items():
            if dir_path.exists():
                files = list(dir_path.glob('*'))
                report['files'][dir_name] = {
                    'count': len(files),
                    'files': [f.name for f in files[:10]],  # First 10
                }
        
        return report
    
    def print_validation_report(self, report: Dict[str, Any]) -> None:
        """
        Print validation report.
        
        Args:
            report: Validation report dictionary
        """
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        
        print(f"Timestamp: {report['timestamp']}")
        print(f"Status: {'PASSED' if report['validation_passed'] else 'FAILED'}")
        
        print("\nDirectories:")
        for dir_name, info in report['directories'].items():
            status = "✓" if info['exists'] else "✗"
            print(f"  {status} {dir_name}: {info['path']}")
        
        print("\nFiles:")
        for dir_name, info in report['files'].items():
            print(f"  {dir_name}: {info['count']} files")
            if info['files']:
                for fname in info['files'][:3]:
                    print(f"    - {fname}")
        
        print("="*60 + "\n")

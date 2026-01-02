"""
Output Validation and Verification Module.

Validates:
- JSON schema compliance
- File existence and integrity
- Data consistency
- Output completeness
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OutputValidator:
    """
    Validates all outputs against expected schemas and requirements.
    """
    
    @staticmethod
    def validate_json_prediction(json_path: str) -> Tuple[bool, List[str]]:
        """
        Validate a single JSON prediction file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            return False, [f"Failed to load JSON: {str(e)}"]
        
        # Required top-level keys
        required_keys = ['prediction', 'confidence', 'explanation', 'context', 'metadata']
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing top-level key: {key}")
        
        # Validate prediction
        if 'prediction' in data:
            pred = data['prediction']
            if 'flood_probability' not in pred:
                errors.append("Missing prediction.flood_probability")
            elif not isinstance(pred['flood_probability'], (int, float)):
                errors.append("prediction.flood_probability must be numeric")
            elif not (0 <= pred['flood_probability'] <= 1):
                errors.append("prediction.flood_probability must be between 0 and 1")
            
            if 'risk_level' not in pred:
                errors.append("Missing prediction.risk_level")
            elif pred['risk_level'] not in ['LOW', 'MEDIUM', 'HIGH']:
                errors.append(f"Invalid risk_level: {pred['risk_level']}")
        
        # Validate confidence
        if 'confidence' in data:
            conf = data['confidence']
            if 'model_confidence' not in conf:
                errors.append("Missing confidence.model_confidence")
            elif not isinstance(conf['model_confidence'], (int, float)):
                errors.append("confidence.model_confidence must be numeric")
            elif not (0 <= conf['model_confidence'] <= 1):
                errors.append("confidence.model_confidence must be between 0 and 1")
        
        # Validate explanation
        if 'explanation' in data:
            if not isinstance(data['explanation'], list):
                errors.append("explanation must be a list")
            elif len(data['explanation']) == 0:
                errors.append("explanation list is empty")
            else:
                for i, exp in enumerate(data['explanation']):
                    if not isinstance(exp, str):
                        errors.append(f"explanation[{i}] must be string")
        
        # Validate context
        if 'context' in data:
            ctx = data['context']
            if 'location' not in ctx:
                errors.append("Missing context.location")
            if 'time_window' not in ctx:
                errors.append("Missing context.time_window")
            
            if 'location' in ctx:
                loc = ctx['location']
                required_loc = ['district', 'state', 'latitude', 'longitude']
                for key in required_loc:
                    if key not in loc:
                        errors.append(f"Missing context.location.{key}")
            
            if 'time_window' in ctx:
                tw = ctx['time_window']
                if 'prediction_time' not in tw:
                    errors.append("Missing context.time_window.prediction_time")
                if 'valid_until' not in tw:
                    errors.append("Missing context.time_window.valid_until")
        
        # Validate metadata
        if 'metadata' in data:
            meta = data['metadata']
            required_meta = ['model_type', 'model_version', 'data_source', 'disclaimer']
            for key in required_meta:
                if key not in meta:
                    errors.append(f"Missing metadata.{key}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_all_json_predictions(json_dir: str) -> Dict[str, any]:
        """
        Validate all JSON prediction files in directory.
        
        Args:
            json_dir: Directory containing JSON files
            
        Returns:
            Validation report dictionary
        """
        json_path = Path(json_dir)
        
        if not json_path.exists():
            logger.error(f"JSON directory not found: {json_dir}")
            return {
                'valid': False,
                'error': f"Directory not found: {json_dir}",
                'total_files': 0,
                'valid_files': 0,
                'invalid_files': 0,
            }
        
        json_files = list(json_path.glob('predictions_*.json'))
        
        valid_count = 0
        invalid_count = 0
        errors_by_file = {}
        
        for json_file in json_files:
            is_valid, errors = OutputValidator.validate_json_prediction(str(json_file))
            
            if is_valid:
                valid_count += 1
                logger.debug(f"✓ Valid: {json_file.name}")
            else:
                invalid_count += 1
                errors_by_file[json_file.name] = errors
                logger.warning(f"✗ Invalid: {json_file.name}")
                for error in errors:
                    logger.warning(f"  - {error}")
        
        report = {
            'valid': invalid_count == 0,
            'total_files': len(json_files),
            'valid_files': valid_count,
            'invalid_files': invalid_count,
            'errors_by_file': errors_by_file,
        }
        
        return report
    
    @staticmethod
    def validate_csv_files(csv_dir: str) -> Dict[str, any]:
        """
        Validate CSV files in directory.
        
        Args:
            csv_dir: Directory containing CSV files
            
        Returns:
            Validation report dictionary
        """
        csv_path = Path(csv_dir)
        
        if not csv_path.exists():
            return {
                'valid': False,
                'error': f"Directory not found: {csv_dir}",
                'total_files': 0,
            }
        
        csv_files = list(csv_path.glob('*.csv'))
        
        report = {
            'valid': True,
            'total_files': len(csv_files),
            'files': {},
        }
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                report['files'][csv_file.name] = {
                    'valid': True,
                    'records': len(df),
                    'columns': len(df.columns),
                }
            except Exception as e:
                report['valid'] = False
                report['files'][csv_file.name] = {
                    'valid': False,
                    'error': str(e),
                }
        
        return report
    
    @staticmethod
    def print_validation_summary(json_report: Dict, csv_report: Dict) -> None:
        """
        Print validation summary.
        
        Args:
            json_report: JSON validation report
            csv_report: CSV validation report
        """
        print("\n" + "="*70)
        print("OUTPUT VALIDATION SUMMARY")
        print("="*70)
        
        print("\nJSON PREDICTIONS:")
        print(f"  Total files: {json_report['total_files']}")
        print(f"  Valid files: {json_report['valid_files']}")
        print(f"  Invalid files: {json_report['invalid_files']}")
        
        if json_report['invalid_files'] > 0:
            print("\n  Invalid files:")
            for filename, errors in json_report['errors_by_file'].items():
                print(f"    - {filename}")
                for error in errors[:3]:  # Show first 3 errors
                    print(f"      • {error}")
        
        print("\nCSV FILES:")
        print(f"  Total files: {csv_report['total_files']}")
        
        for filename, info in csv_report['files'].items():
            if info['valid']:
                print(f"  ✓ {filename}: {info['records']} records, {info['columns']} columns")
            else:
                print(f"  ✗ {filename}: {info['error']}")
        
        print("\nOVERALL STATUS:")
        if json_report['valid'] and csv_report['valid']:
            print("  ✓ ALL OUTPUTS VALID")
        else:
            print("  ✗ VALIDATION FAILED")
        
        print("="*70 + "\n")

"""
Flood Early Warning Prediction System - Main Entry Point.

Complete end-to-end pipeline: Data → Clean → Features → Train → Predict → Alerts

Data Source: Central Water Commission (CWC), Government of India
Website: https://ffs.india-water.gov.in/

Usage:
    python main.py                    # Run complete pipeline
    python main.py --data-file <path> # Use specific data file
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import sys
import argparse
from typing import Dict

from src.utils.logger import setup_logger
from src.utils.constants import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FEATURE_NAMES
from src.utils.output_manager import OutputManager
from src.data_engineering import DataCleaner
from src.features import FeatureGenerator
from src.models import ModelTrainer
from src.inference import FloodPredictor, AlertGenerator
from src.explainability import ExplanationGenerator

logger = setup_logger(__name__)


class CompleteFloodPredictionPipeline:
    """
    Complete end-to-end flood prediction pipeline with explicit output management.
    """
    
    def __init__(self):
        """Initialize pipeline."""
        self.cleaner = DataCleaner()
        self.feature_gen = FeatureGenerator()
        self.trainer = ModelTrainer()
        self.predictor = FloodPredictor()
        self.alert_gen = AlertGenerator()
        self.explainer = ExplanationGenerator()
        self.output_manager = OutputManager()
        logger.info("Complete Pipeline initialized")
    
    def load_raw_data(self, csv_path: str) -> pd.DataFrame:
        """Load raw CWC data from CSV."""
        logger.info(f"Loading raw data from {csv_path}...")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: DATA CLEANING")
        logger.info("="*70)
        
        cleaned = self.cleaner.clean(raw_data)
        logger.info(f"Cleaned data shape: {cleaned.shape}")
        
        # Save cleaned data
        cleaned_path = self.output_manager.save_cleaned_data(cleaned)
        
        self.output_manager.print_pipeline_summary(
            "Data Cleaning",
            {
                'Cleaned records': len(cleaned),
                'Saved to': cleaned_path,
            }
        )
        
        return cleaned
    
    def generate_features(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """Generate ML features."""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*70)
        
        # Prepare data for feature generator
        data_dict = {
            'stations': cleaned_data[['station_id', 'station_name', 'river_name', 'basin', 'state']].drop_duplicates(),
            'levels': cleaned_data
        }
        
        features = self.feature_gen.generate(data_dict)
        logger.info(f"Generated features shape: {features.shape}")
        logger.info(f"Feature columns: {[col for col in FEATURE_NAMES if col in features.columns]}")
        
        # Save features
        features_path = self.output_manager.save_features(features)
        
        self.output_manager.print_pipeline_summary(
            "Feature Engineering",
            {
                'Feature records': len(features),
                'Feature count': len(features.columns),
                'Saved to': features_path,
            }
        )
        
        return features
    
    def create_labels(self, features: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic labels for training."""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: CREATING TRAINING LABELS")
        logger.info("="*70)
        
        # Create labels based on distance to danger level
        labels = pd.DataFrame()
        
        # Risk score: based on distance to danger level
        distance_to_danger = features.get('distance_to_danger', 0)
        labels['risk_score'] = np.clip(1.0 - (distance_to_danger / 5.0), 0, 1)
        
        # Binary warning: if close to danger level (create some positive cases)
        # Use a threshold that creates ~20% positive cases
        threshold = distance_to_danger.quantile(0.2)
        labels['warning_label'] = (distance_to_danger < threshold).astype(int)
        
        # Risk tier: low/medium/high
        labels['risk_tier'] = pd.cut(labels['risk_score'], 
                                     bins=[0, 0.33, 0.67, 1.0],
                                     labels=[0, 1, 2]).astype(int)
        
        logger.info(f"Created labels: {labels.shape}")
        logger.info(f"Risk score range: {labels['risk_score'].min():.3f} - {labels['risk_score'].max():.3f}")
        logger.info(f"Warning labels: {labels['warning_label'].sum()} positive ({labels['warning_label'].mean()*100:.1f}%)")
        logger.info(f"Risk tiers: {labels['risk_tier'].value_counts().to_dict()}")
        
        return labels
    
    def train_models(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict:
        """Train prediction models."""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("="*70)
        
        from sklearn.model_selection import train_test_split
        
        # Split data consistently
        X_train, X_test, y_train_risk, y_test_risk = train_test_split(
            features, labels['risk_score'], test_size=0.2, random_state=42
        )
        _, _, y_train_warn, y_test_warn = train_test_split(
            features, labels['warning_label'], test_size=0.2, random_state=42
        )
        _, _, y_train_tier, y_test_tier = train_test_split(
            features, labels['risk_tier'], test_size=0.2, random_state=42
        )
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[[col for col in FEATURE_NAMES if col in X_train.columns]])
        X_test_scaled = scaler.transform(X_test[[col for col in FEATURE_NAMES if col in X_test.columns]])
        
        X_train = pd.DataFrame(X_train_scaled, columns=[col for col in FEATURE_NAMES if col in X_train.columns])
        X_test = pd.DataFrame(X_test_scaled, columns=[col for col in FEATURE_NAMES if col in X_test.columns])
        
        results = {}
        
        # Train risk regressor
        logger.info("\nTraining Risk Regressor...")
        risk_results = self.trainer.train_risk_regressor(
            X_train, y_train_risk,
            X_test, y_test_risk
        )
        results['risk_regressor'] = risk_results['metrics']
        logger.info(f"Risk Regressor - Test RMSE: {risk_results['metrics']['test_rmse']:.4f}")
        
        # Train warning classifier
        logger.info("\nTraining Warning Classifier...")
        warning_results = self.trainer.train_warning_classifier(
            X_train, y_train_warn,
            X_test, y_test_warn
        )
        results['warning_classifier'] = warning_results['metrics']
        logger.info(f"Warning Classifier - Recall: {warning_results['metrics']['recall']:.4f}")
        
        # Train risk tier classifier
        logger.info("\nTraining Risk Tier Classifier...")
        tier_results = self.trainer.train_risk_tier_classifier(
            X_train, y_train_tier,
            X_test, y_test_tier
        )
        results['risk_tier_classifier'] = tier_results['metrics']
        logger.info(f"Risk Tier Classifier - Macro F1: {tier_results['metrics']['macro_f1']:.4f}")
        
        # Save models
        self.trainer.save_models(str(MODELS_DIR))
        logger.info(f"\nModels saved to {MODELS_DIR}")
        
        return results
    
    def generate_predictions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using trained models."""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: INFERENCE")
        logger.info("="*70)
        
        # Reload predictor to get trained models
        self.predictor = FloodPredictor(model_dir=str(MODELS_DIR))
        
        predictions = self.predictor.predict_all(features)
        logger.info(f"Generated predictions: {predictions.shape}")
        
        # Save predictions table
        predictions_table_path = self.output_manager.save_predictions_table(predictions)
        
        self.output_manager.print_pipeline_summary(
            "Predictions Generated",
            {
                'Prediction records': len(predictions),
                'Saved to': predictions_table_path,
            }
        )
        
        return predictions
    
    def generate_alerts(self, predictions: pd.DataFrame, 
                       station_metadata: pd.DataFrame) -> pd.DataFrame:
        """Generate alerts from predictions."""
        logger.info("\n" + "="*70)
        logger.info("STEP 6: ALERT GENERATION")
        logger.info("="*70)
        
        alerts = self.alert_gen.generate_alerts(predictions, station_metadata)
        logger.info(f"Generated {len(alerts)} alerts")
        
        if len(alerts) > 0:
            summary = self.alert_gen.get_alert_summary(alerts)
            logger.info(f"Alert Summary: {summary}")
        
        return alerts
    
    def run(self, csv_path: str = 'data/raw/cwc_hydrograph_20260102_050433.csv') -> Dict:
        """Run complete pipeline."""
        logger.info("\n" + "="*70)
        logger.info("FLOOD EARLY WARNING PREDICTION SYSTEM")
        logger.info("="*70)
        logger.info(f"Data Source: Central Water Commission (CWC), Government of India")
        logger.info(f"Website: https://ffs.india-water.gov.in/")
        logger.info("="*70 + "\n")
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'failed'
        }
        
        try:
            # Load data
            raw_data = self.load_raw_data(csv_path)
            results['raw_records'] = len(raw_data)
            
            # Save raw data
            raw_path = self.output_manager.save_raw_data(
                raw_data,
                source_url='https://ffs.india-water.gov.in/',
                scraper_version='1.0'
            )
            
            self.output_manager.print_pipeline_summary(
                "Raw Data Loaded",
                {
                    'Records': len(raw_data),
                    'Saved to': raw_path,
                }
            )
            
            # Clean data
            cleaned_data = self.clean_data(raw_data)
            results['cleaned_records'] = len(cleaned_data)
            
            # Generate features
            features = self.generate_features(cleaned_data)
            results['feature_records'] = len(features)
            
            # Create labels
            labels = self.create_labels(features)
            
            # Train models
            training_results = self.train_models(features, labels)
            results['training_metrics'] = training_results
            
            # Generate predictions
            predictions = self.generate_predictions(features)
            results['prediction_records'] = len(predictions)
            
            # Get station metadata (include latitude, longitude, district)
            station_metadata = cleaned_data[['station_id', 'station_name', 'river_name', 'basin', 'state', 'latitude', 'longitude', 'district']].drop_duplicates()
            
            # Generate alerts
            alerts = self.generate_alerts(predictions, station_metadata)
            results['alert_records'] = len(alerts)
            
            # Save results (including JSON conversion)
            self.save_results(predictions, alerts, results, station_metadata)
            
            results['status'] = 'success'
            
            # Print summary
            logger.info("\n" + "="*70)
            logger.info("PIPELINE EXECUTION COMPLETE")
            logger.info("="*70)
            logger.info(f"Raw Records: {results['raw_records']}")
            logger.info(f"Cleaned Records: {results['cleaned_records']}")
            logger.info(f"Features Generated: {results['feature_records']}")
            logger.info(f"Predictions: {results['prediction_records']}")
            logger.info(f"Alerts: {results['alert_records']}")
            logger.info("="*70 + "\n")
            
            # Validate outputs
            validation_report = self.output_manager.validate_outputs()
            self.output_manager.print_validation_report(validation_report)
            
            return results
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def save_results(self, predictions: pd.DataFrame, alerts: pd.DataFrame, 
                    results: Dict, station_metadata: pd.DataFrame) -> None:
        """Save pipeline results including JSON predictions."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions table (already done in generate_predictions)
        # Now convert to JSON with filtering
        logger.info("\n" + "="*70)
        logger.info("STEP 6: JSON CONVERSION (FILTERED)")
        logger.info("="*70)
        
        json_stats = self.output_manager.save_all_predictions_json(
            predictions,
            station_metadata
        )
        
        self.output_manager.print_pipeline_summary(
            "JSON Predictions (Filtered)",
            {
                'Total stations processed': json_stats['total_processed'],
                'MEDIUM risk stations': json_stats['medium_risk_count'],
                'HIGH risk stations': json_stats['high_risk_count'],
                'LOW risk stations': json_stats['low_risk_count'],
                'LOW risk JSON generated': json_stats['low_risk_json_generated'],
                'JSON files generated': json_stats['json_generated'],
                'Location': str(self.output_manager.dirs['outputs_json']),
            }
        )
        
        # Store stats for final summary
        results['json_stats'] = json_stats
        
        # Save alerts if any
        if not alerts.empty:
            alert_file = PROCESSED_DATA_DIR / f"alerts_{timestamp}.csv"
            alerts.to_csv(alert_file, index=False)
            logger.info(f"Saved alerts to {alert_file}")
        
        # Save results metadata
        results_file = PROCESSED_DATA_DIR / f"results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {results_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Flood Early Warning Prediction System - Complete Pipeline"
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='Path to CWC hydrograph CSV file (auto-detects latest if not specified)'
    )
    
    args = parser.parse_args()
    
    # Find data file
    if args.data_file:
        data_file = args.data_file
        if not Path(data_file).exists():
            logger.error(f"Data file not found: {data_file}")
            return 1
    else:
        # Find latest CWC data file
        raw_files = list(RAW_DATA_DIR.glob('cwc_hydrograph_*.csv'))
        
        if not raw_files:
            logger.error("No CWC hydrograph data found. Run scripts/scrape_and_process.py first.")
            return 1
        
        # Use latest file
        data_file = str(sorted(raw_files)[-1])
    
    logger.info(f"Using data file: {data_file}")
    
    # Run pipeline
    pipeline = CompleteFloodPredictionPipeline()
    results = pipeline.run(data_file)
    
    # Validate outputs
    from src.utils.validation import OutputValidator
    
    output_manager = pipeline.output_manager
    
    json_report = OutputValidator.validate_all_json_predictions(
        str(output_manager.dirs['outputs_json'])
    )
    
    csv_report = OutputValidator.validate_csv_files(
        str(output_manager.dirs['outputs_tabular'])
    )
    
    OutputValidator.print_validation_summary(json_report, csv_report)
    
    if results['status'] == 'success':
        logger.info("Pipeline completed successfully")
        
        # Print JSON output summary (CHANGE 5)
        json_stats = results.get('json_stats', {})
        
        print("\n" + "-"*60)
        print("JSON OUTPUT SUMMARY")
        print("-"*60)
        print(f"Total stations processed: {json_stats.get('total_processed', 0)}")
        print(f"Stations with MEDIUM risk: {json_stats.get('medium_risk_count', 0)}")
        print(f"Stations with HIGH risk: {json_stats.get('high_risk_count', 0)}")
        print(f"JSON files generated: {json_stats.get('json_generated', 0)}")
        print(f"JSON files skipped (LOW risk): {json_stats.get('json_skipped', 0)}")
        print(f"JSON directory: data/outputs/json_predictions/")
        print("-"*60 + "\n")
        
        # Print final summary
        print("\n" + "="*70)
        print("FINAL EXECUTION SUMMARY")
        print("="*70)
        print(f"Status: SUCCESS")
        print(f"Raw records: {results.get('raw_records', 0)}")
        print(f"Cleaned records: {results.get('cleaned_records', 0)}")
        print(f"Features generated: {results.get('feature_records', 0)}")
        print(f"Predictions: {results.get('prediction_records', 0)}")
        print(f"JSON files: {json_stats.get('json_generated', 0)}")
        print(f"Alerts: {results.get('alert_records', 0)}")
        print("="*70 + "\n")
        
        return 0
    else:
        logger.error(f"Pipeline failed: {results.get('error')}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""
Main Pipeline Orchestration for Flood Early Warning System.

Data Source: Central Water Commission (CWC), Government of India
Website: https://inf.cwc.gov.in/

Orchestrates the complete workflow:
1. Data Acquisition
2. Data Engineering
3. Feature Engineering
4. Model Training
5. Inference & Alert Generation
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import json

from src.data_acquisition import CWCScraper
from src.data_engineering import DataCleaner
from src.features import FeatureGenerator
from src.models import ModelTrainer
from src.inference import FloodPredictor, AlertGenerator
from src.explainability import ExplanationGenerator, FeatureImportance
from src.utils.logger import setup_logger
from src.utils.constants import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR,
    CWC_ATTRIBUTION, LOOKBACK_HOURS, RATE_OF_RISE_WINDOW_HOURS
)

logger = setup_logger(__name__)


class FloodWarningPipeline:
    """
    Complete flood early warning prediction pipeline.
    
    Handles:
    - Data acquisition from CWC
    - Data cleaning and validation
    - Feature engineering
    - Model training
    - Batch and real-time inference
    - Alert generation
    """
    
    def __init__(self, model_dir: str = str(MODELS_DIR)):
        """
        Initialize pipeline.
        
        Args:
            model_dir: Directory for model artifacts
        """
        self.model_dir = model_dir
        
        # Initialize components
        self.scraper = CWCScraper()
        self.cleaner = DataCleaner()
        self.feature_generator = FeatureGenerator(
            lookback_hours=LOOKBACK_HOURS,
            rate_window_hours=RATE_OF_RISE_WINDOW_HOURS
        )
        self.model_trainer = ModelTrainer()
        self.predictor = FloodPredictor(model_dir=model_dir)
        self.alert_generator = AlertGenerator()
        self.explanation_generator = ExplanationGenerator(model_dir=model_dir)
        self.feature_importance = FeatureImportance(model_dir=model_dir)
        
        logger.info("Flood Warning Pipeline initialized")
        logger.info(f"Data Source: {CWC_ATTRIBUTION['source']}")
        logger.info(f"Attribution: {CWC_ATTRIBUTION['url']}")
    
    def fetch_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch raw data from CWC.
        
        Returns:
            DataFrame with raw station data or None
        """
        logger.info("="*60)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("="*60)
        
        raw_data = self.scraper.fetch_all_stations()
        
        if raw_data is None:
            logger.error("Failed to fetch data from CWC")
            return None
        
        logger.info(f"Fetched {len(raw_data)} station records")
        
        return raw_data
    
    def clean_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate raw data.
        
        Args:
            raw_data: Raw DataFrame from CWC
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("="*60)
        logger.info("STEP 2: DATA ENGINEERING")
        logger.info("="*60)
        
        cleaned_data = self.cleaner.clean(raw_data)
        
        logger.info(f"Cleaned data shape: {cleaned_data.shape}")
        logger.info(f"Validation report: {self.cleaner.get_validation_report()}")
        
        return cleaned_data
    
    def generate_features(self, cleaned_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ML features from cleaned data.
        
        Args:
            cleaned_data: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("="*60)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("="*60)
        
        # Prepare data structure for feature generator
        data_dict = {
            'stations': cleaned_data[['station_id', 'station_name', 'river_name', 'basin', 'state']].drop_duplicates(),
            'levels': cleaned_data[['station_id', 'current_level', 'warning_level', 'danger_level', 'trend', 'timestamp']]
        }
        
        features = self.feature_generator.generate(data_dict)
        
        logger.info(f"Generated features shape: {features.shape}")
        
        return features
    
    def train_models(self, features: pd.DataFrame, labels: pd.DataFrame) -> Dict:
        """
        Train all prediction models.
        
        Args:
            features: Feature DataFrame
            labels: Labels DataFrame with columns: risk_score, warning_label, risk_tier
            
        Returns:
            Dictionary with training results
        """
        logger.info("="*60)
        logger.info("STEP 4: MODEL TRAINING")
        logger.info("="*60)
        
        # Prepare data
        data, metadata = self.model_trainer.prepare_data(features, labels)
        
        results = {}
        
        # Train risk regressor
        logger.info("Training risk regressor...")
        risk_results = self.model_trainer.train_risk_regressor(
            data['X_train'], labels['risk_score'],
            data['X_test'], labels['risk_score']
        )
        results['risk_regressor'] = risk_results['metrics']
        
        # Train warning classifier
        logger.info("Training warning classifier...")
        warning_results = self.model_trainer.train_warning_classifier(
            data['X_train'], labels['warning_label'],
            data['X_test'], labels['warning_label']
        )
        results['warning_classifier'] = warning_results['metrics']
        
        # Train risk tier classifier
        logger.info("Training risk tier classifier...")
        tier_results = self.model_trainer.train_risk_tier_classifier(
            data['X_train'], labels['risk_tier'],
            data['X_test'], labels['risk_tier']
        )
        results['risk_tier_classifier'] = tier_results['metrics']
        
        # Save models
        self.model_trainer.save_models(self.model_dir)
        
        logger.info("Model training complete")
        
        return results
    
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for features.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            DataFrame with predictions
        """
        logger.info("="*60)
        logger.info("STEP 5: INFERENCE")
        logger.info("="*60)
        
        predictions = self.predictor.predict_all(features)
        
        logger.info(f"Generated predictions for {len(predictions)} records")
        
        return predictions
    
    def generate_alerts(self, predictions: pd.DataFrame,
                       station_metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate alerts from predictions.
        
        Args:
            predictions: Predictions DataFrame
            station_metadata: Optional station metadata
            
        Returns:
            DataFrame with alerts
        """
        logger.info("="*60)
        logger.info("STEP 6: ALERT GENERATION")
        logger.info("="*60)
        
        alerts = self.alert_generator.generate_alerts(predictions, station_metadata)
        
        summary = self.alert_generator.get_alert_summary(alerts)
        logger.info(f"Alert summary: {summary}")
        
        return alerts
    
    def run_full_pipeline(self, train_models: bool = False,
                         labels: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run complete pipeline from data acquisition to alerts.
        
        Args:
            train_models: Whether to train models
            labels: Optional labels for training
            
        Returns:
            Dictionary with pipeline results
        """
        logger.info("\n" + "="*60)
        logger.info("FLOOD EARLY WARNING PREDICTION SYSTEM")
        logger.info("="*60)
        logger.info(f"Data Source: {CWC_ATTRIBUTION['source']}")
        logger.info(f"Timestamp: {datetime.utcnow().isoformat()}")
        logger.info("="*60 + "\n")
        
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'data_source': CWC_ATTRIBUTION['source'],
            'disclaimer': CWC_ATTRIBUTION['disclaimer'],
        }
        
        # Step 1: Fetch data
        raw_data = self.fetch_data()
        if raw_data is None:
            logger.error("Pipeline failed at data acquisition")
            return results
        
        results['raw_data_count'] = len(raw_data)
        
        # Step 2: Clean data
        cleaned_data = self.clean_data(raw_data)
        results['cleaned_data_count'] = len(cleaned_data)
        
        # Step 3: Generate features
        features = self.generate_features(cleaned_data)
        results['features_count'] = len(features)
        
        # Step 4: Train models (optional)
        if train_models and labels is not None:
            training_results = self.train_models(features, labels)
            results['training_metrics'] = training_results
        
        # Step 5: Generate predictions
        predictions = self.predict(features)
        results['predictions_count'] = len(predictions)
        
        # Step 6: Generate alerts
        station_metadata = cleaned_data[['station_id', 'station_name', 'river_name', 'state']].drop_duplicates()
        alerts = self.generate_alerts(predictions, station_metadata)
        results['alerts_count'] = len(alerts)
        results['alert_summary'] = self.alert_generator.get_alert_summary(alerts)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Results: {json.dumps(results, indent=2, default=str)}")
        logger.info("="*60 + "\n")
        
        return results
    
    def run_inference_only(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run inference-only pipeline (no training).
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Tuple of (predictions, alerts)
        """
        logger.info("Running inference-only pipeline...")
        
        predictions = self.predict(features)
        alerts = self.generate_alerts(predictions)
        
        return predictions, alerts
    
    def get_model_status(self) -> Dict:
        """Get status of all models."""
        return self.predictor.get_model_status()
    
    def get_feature_importance(self, model_name: str, top_n: int = 5) -> pd.DataFrame:
        """Get feature importance for a model."""
        self.feature_importance.load_models()
        return self.feature_importance.get_feature_importance(model_name, top_n)
    
    def explain_prediction(self, features: pd.Series, risk_score: float,
                          warning_prob: float, risk_tier: str) -> Dict:
        """Get explanation for a prediction."""
        self.explanation_generator.load_models()
        
        explanation = {
            'risk_score': self.explanation_generator.explain_risk_score(features, risk_score),
            'warning': self.explanation_generator.explain_warning_prediction(features, warning_prob),
            'risk_tier': self.explanation_generator.explain_risk_tier(features, risk_tier),
        }
        
        return explanation

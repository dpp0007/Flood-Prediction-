"""
Example Usage of Flood Early Warning System.

Data Source: Central Water Commission (CWC), Government of India
Website: https://inf.cwc.gov.in/

Demonstrates:
1. Data acquisition
2. Data cleaning
3. Feature engineering
4. Model training
5. Inference
6. Alert generation
7. Explanations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pipeline import FloodWarningPipeline
from src.data_acquisition import CWCScraper
from src.data_engineering import DataCleaner
from src.features import FeatureGenerator
from src.models import ModelTrainer
from src.inference import FloodPredictor, AlertGenerator
from src.explainability import ExplanationGenerator, FeatureImportance
from src.utils.logger import setup_logger
from src.utils.constants import FEATURE_NAMES

logger = setup_logger(__name__)


def example_1_data_acquisition():
    """Example 1: Fetch data from CWC."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: DATA ACQUISITION")
    logger.info("="*60)
    
    scraper = CWCScraper()
    
    # Fetch station data
    station_data = scraper.fetch_station_data()
    
    if station_data is not None:
        logger.info(f"Fetched {len(station_data)} stations")
        logger.info(f"Columns: {station_data.columns.tolist()}")
        logger.info(f"\nFirst few records:\n{station_data.head()}")
    else:
        logger.warning("No data fetched. CWC website may be unavailable.")


def example_2_data_cleaning():
    """Example 2: Clean and validate data."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: DATA CLEANING")
    logger.info("="*60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'station_id': ['S001', 'S002', 'S003', 'S001', 'S002'],
        'station_name': ['Station A', 'Station B', 'Station C', 'Station A', 'Station B'],
        'current_level': [5.2, 3.1, np.nan, 5.5, 3.0],
        'warning_level': [7.0, 5.0, 6.0, 7.0, 5.0],
        'danger_level': [8.0, 6.0, 7.0, 8.0, 6.0],
        'timestamp': [datetime.utcnow() - timedelta(hours=i) for i in range(5)],
        'trend': ['rising', 'falling', 'stable', 'rising', 'falling']
    })
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(sample_data)
    
    logger.info(f"Original shape: {sample_data.shape}")
    logger.info(f"Cleaned shape: {cleaned_data.shape}")
    logger.info(f"Validation report: {cleaner.get_validation_report()}")


def example_3_feature_engineering():
    """Example 3: Generate features."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: FEATURE ENGINEERING")
    logger.info("="*60)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'station_id': ['S001', 'S002', 'S001', 'S002'],
        'station_name': ['Station A', 'Station B', 'Station A', 'Station B'],
        'current_level': [5.2, 3.1, 5.5, 3.0],
        'warning_level': [7.0, 5.0, 7.0, 5.0],
        'danger_level': [8.0, 6.0, 8.0, 6.0],
        'observation_time': [datetime.utcnow() - timedelta(hours=i) for i in range(4)],
        'trend': ['rising', 'falling', 'rising', 'falling']
    })
    
    # Prepare data structure
    data_dict = {
        'stations': sample_data[['station_id', 'station_name']].drop_duplicates(),
        'levels': sample_data
    }
    
    generator = FeatureGenerator()
    features = generator.generate(data_dict)
    
    logger.info(f"Generated features shape: {features.shape}")
    logger.info(f"Feature columns: {[col for col in FEATURE_NAMES if col in features.columns]}")


def example_4_model_training():
    """Example 4: Train models."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: MODEL TRAINING")
    logger.info("="*60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Create sample features and labels
    n_samples = 100
    features = pd.DataFrame({
        col: np.random.randn(n_samples) for col in FEATURE_NAMES
    })
    
    # Create labels with matching sample count
    risk_scores = np.random.uniform(0, 1, n_samples)
    warning_labels = np.random.randint(0, 2, n_samples)
    risk_tiers = np.random.randint(0, 3, n_samples)
    
    # Split data consistently
    X_train, X_test, y_train_risk, y_test_risk = train_test_split(
        features, risk_scores, test_size=0.2, random_state=42
    )
    _, _, y_train_warn, y_test_warn = train_test_split(
        features, warning_labels, test_size=0.2, random_state=42
    )
    _, _, y_train_tier, y_test_tier = train_test_split(
        features, risk_tiers, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train_scaled, columns=FEATURE_NAMES)
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURE_NAMES)
    
    trainer = ModelTrainer()
    
    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Train models
    logger.info("Training risk regressor...")
    risk_results = trainer.train_risk_regressor(
        X_train, y_train_risk,
        X_test, y_test_risk
    )
    logger.info(f"Risk regressor metrics: {risk_results['metrics']}")
    
    logger.info("Training warning classifier...")
    warning_results = trainer.train_warning_classifier(
        X_train, y_train_warn,
        X_test, y_test_warn
    )
    logger.info(f"Warning classifier metrics: {warning_results['metrics']}")


def example_5_inference():
    """Example 5: Generate predictions."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: INFERENCE")
    logger.info("="*60)
    
    # Create sample features
    n_samples = 10
    features = pd.DataFrame({
        'station_id': [f'S{i:03d}' for i in range(n_samples)],
        'observation_time': [datetime.utcnow()] * n_samples,
        **{col: np.random.randn(n_samples) for col in FEATURE_NAMES}
    })
    
    predictor = FloodPredictor()
    
    # Check model status
    status = predictor.get_model_status()
    logger.info(f"Model status: {status}")
    
    if not all(status.values()):
        logger.warning("Models not loaded. Skipping inference.")
        return
    
    # Generate predictions
    predictions = predictor.predict_all(features)
    
    logger.info(f"Generated predictions for {len(predictions)} records")
    logger.info(f"\nPredictions:\n{predictions}")


def example_6_alert_generation():
    """Example 6: Generate alerts."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 6: ALERT GENERATION")
    logger.info("="*60)
    
    # Create sample predictions
    predictions = pd.DataFrame({
        'station_id': ['S001', 'S002', 'S003'],
        'station_name': ['Station A', 'Station B', 'Station C'],
        'risk_score': [0.85, 0.45, 0.25],
        'warning_probability': [0.75, 0.35, 0.15],
        'risk_tier_name': ['high', 'medium', 'low']
    })
    
    alert_gen = AlertGenerator()
    alerts = alert_gen.generate_alerts(predictions)
    
    logger.info(f"Generated {len(alerts)} alerts")
    logger.info(f"\nAlerts:\n{alerts[['station_name', 'alert_severity', 'alert_message']]}")
    
    # Get alert summary
    summary = alert_gen.get_alert_summary(alerts)
    logger.info(f"\nAlert summary: {summary}")


def example_7_explanations():
    """Example 7: Generate explanations."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 7: EXPLANATIONS")
    logger.info("="*60)
    
    # Create sample features
    features = pd.Series({
        col: np.random.randn() for col in FEATURE_NAMES
    })
    
    explainer = ExplanationGenerator()
    explainer.load_models()
    
    # Generate explanations
    risk_explanation = explainer.explain_risk_score(features, risk_score=0.75)
    logger.info(f"Risk score explanation: {risk_explanation}")
    
    warning_explanation = explainer.explain_warning_prediction(features, warning_prob=0.65)
    logger.info(f"Warning explanation: {warning_explanation}")
    
    # Generate alert text
    alert_text = explainer.generate_alert_text(
        station_id='S001',
        station_name='Test Station',
        risk_score=0.75,
        warning_prob=0.65,
        risk_tier='high',
        features=features
    )
    logger.info(f"Alert text:\n{alert_text}")


def example_8_full_pipeline():
    """Example 8: Run full pipeline."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 8: FULL PIPELINE")
    logger.info("="*60)
    
    pipeline = FloodWarningPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(train_models=False)
    
    logger.info(f"Pipeline results: {results}")


def main():
    """Run all examples."""
    logger.info("\n" + "="*70)
    logger.info("FLOOD EARLY WARNING SYSTEM - USAGE EXAMPLES")
    logger.info("="*70)
    logger.info("Data Source: Central Water Commission (CWC), Government of India")
    logger.info("Website: https://inf.cwc.gov.in/")
    logger.info("="*70)
    
    # Run examples
    example_1_data_acquisition()
    example_2_data_cleaning()
    example_3_feature_engineering()
    example_4_model_training()
    example_5_inference()
    example_6_alert_generation()
    example_7_explanations()
    # example_8_full_pipeline()  # Uncomment to run full pipeline
    
    logger.info("\n" + "="*70)
    logger.info("EXAMPLES COMPLETE")
    logger.info("="*70)


if __name__ == '__main__':
    main()

"""
Model training and evaluation for flood prediction

Data Source: Central Water Commission (CWC), Government of India

Trains three predictive layers:
1. Continuous Risk Score (Regression)
2. Binary Early Warning (Classification)
3. Multi-class Risk Tiers (Classification)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib
from typing import Dict, Tuple, Any
from src.utils.logger import setup_logger
from src.utils.constants import FEATURE_NAMES, RISK_TIERS

logger = setup_logger(__name__)

class ModelTrainer:
    """
    Trains and evaluates flood prediction models.
    
    Models:
    - Risk Regressor: Continuous risk score [0, 1]
    - Warning Classifier: Binary early warning
    - Risk Tier Classifier: Multi-class risk levels
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize model trainer."""
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        logger.info("Model trainer initialized")
    
    def prepare_data(self, features_df: pd.DataFrame, labels_df: pd.DataFrame,
                     test_size: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Prepare data for training.
        
        Args:
            features_df: Feature DataFrame
            labels_df: Labels DataFrame
            test_size: Test set ratio
        
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        logger.info("Preparing training data...")
        
        # Select feature columns
        feature_cols = [col for col in FEATURE_NAMES if col in features_df.columns]
        X = features_df[feature_cols].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, labels_df, test_size=test_size, random_state=self.random_state
        )
        
        self.scalers['features'] = scaler
        
        logger.info(f"Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test
        }, {'scaler': scaler}
    
    def train_risk_regressor(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            model_type: str = 'gradient_boosting') -> Dict[str, Any]:
        """
        Train continuous risk score regressor.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_type: 'gradient_boosting' or 'random_forest'
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training risk regressor ({model_type})...")
        
        if model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=8, learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=8,
                random_state=self.random_state
            )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
        }
        
        logger.info(f"Risk Regressor - Test RMSE: {metrics['test_rmse']:.4f}, R²: {metrics['test_r2']:.4f}")
        
        self.models['risk_regressor'] = model
        self.metrics['risk_regressor'] = metrics
        
        return {'model': model, 'metrics': metrics}
    
    def train_warning_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                model_type: str = 'gradient_boosting') -> Dict[str, Any]:
        """
        Train binary early warning classifier.
        
        Args:
            X_train, y_train: Training data (binary labels)
            X_test, y_test: Test data
            model_type: 'gradient_boosting' or 'logistic_regression'
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training warning classifier ({model_type})...")
        
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'recall': recall_score(y_test, y_pred_test, zero_division=0),
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
        }
        
        logger.info(f"Warning Classifier - Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        self.models['warning_classifier'] = model
        self.metrics['warning_classifier'] = metrics
        
        return {'model': model, 'metrics': metrics}
    
    def train_risk_tier_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train multi-class risk tier classifier.
        
        Args:
            X_train, y_train: Training data (multi-class labels)
            X_test, y_test: Test data
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training risk tier classifier...")
        
        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.1,
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        
        metrics = {
            'macro_f1': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        }
        
        logger.info(f"Risk Tier Classifier - Macro F1: {metrics['macro_f1']:.4f}")
        
        self.models['risk_tier_classifier'] = model
        self.metrics['risk_tier_classifier'] = metrics
        
        return {'model': model, 'metrics': metrics}
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """Get feature importance for tree-based models."""
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        importances = model.feature_importances_
        feature_cols = [col for col in FEATURE_NAMES if col in self.scalers.get('features', {}).get_feature_names_out([]) if hasattr(self.scalers.get('features'), 'get_feature_names_out')]
        
        importance_df = pd.DataFrame({
            'feature': feature_cols[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save_models(self, model_dir: str) -> None:
        """Save trained models and scalers."""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for name, model in self.models.items():
            path = os.path.join(model_dir, f"{name}.pkl")
            joblib.dump(model, path)
            logger.info(f"Saved model: {path}")
        
        for name, scaler in self.scalers.items():
            path = os.path.join(model_dir, f"{name}_scaler.pkl")
            joblib.dump(scaler, path)
            logger.info(f"Saved scaler: {path}")
    
    def load_models(self, model_dir: str) -> None:
        """Load trained models and scalers."""
        import os
        
        for model_name in ['risk_regressor', 'warning_classifier', 'risk_tier_classifier']:
            path = os.path.join(model_dir, f"{model_name}.pkl")
            if os.path.exists(path):
                self.models[model_name] = joblib.load(path)
                logger.info(f"Loaded model: {path}")
        
        scaler_path = os.path.join(model_dir, "features_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scalers['features'] = joblib.load(scaler_path)
            logger.info(f"Loaded scaler: {scaler_path}")

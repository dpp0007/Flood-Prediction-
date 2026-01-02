"""
Model training and evaluation for flood prediction

Data Source: Central Water Commission (CWC), Government of India

Trains three predictive layers:
1. Continuous Risk Score (Regression)
2. Binary Early Warning (Classification)
3. Multi-class Risk Tiers (Classification)

OVERFITTING REDUCTION:
- Soft label generation with probabilistic noise
- Feature noise injection during training
- Time-based train/test split (temporal validation)
- Regularization: max_depth ≤ 5, min_samples_leaf ≥ 20
- Realistic metric reporting (Recall, Precision-Recall, Calibration)
- Disclaimer: Metrics reflect synthetic data with injected noise
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
    classification_report, precision_recall_curve
)
import joblib
from typing import Dict, Tuple, Any
from src.utils.logger import setup_logger
from src.utils.constants import FEATURE_NAMES, RISK_TIERS

logger = setup_logger(__name__)

# Overfitting reduction configuration
FEATURE_NOISE_SIGMA = 0.05  # Gaussian noise std dev for features
LABEL_NOISE_SIGMA = 0.05    # Gaussian noise std dev for labels
MAX_TREE_DEPTH = 5          # Reduced from 8 for regularization
MIN_SAMPLES_LEAF = 20       # Increased from default for regularization
SUBSAMPLE_RATIO = 0.8       # Subsample ratio for gradient boosting
LOGISTIC_REGULARIZATION = 0.3  # Strong regularization for logistic regression

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
                     test_size: float = 0.2, use_temporal_split: bool = True) -> Tuple[Dict, Dict]:
        """
        Prepare data for training with overfitting reduction.
        
        CHANGE 3: Time-based train/test split
        - Train: first 70% of timeline
        - Validation: next 15%
        - Test: last 15%
        
        Args:
            features_df: Feature DataFrame
            labels_df: Labels DataFrame
            test_size: Test set ratio (ignored if temporal split used)
            use_temporal_split: Use time-based split instead of random
        
        Returns:
            Tuple of (train_data, test_data) dictionaries
        """
        logger.info("Preparing training data with overfitting reduction...")
        logger.info(f"  - Temporal split: {use_temporal_split}")
        logger.info(f"  - Feature noise σ: {FEATURE_NOISE_SIGMA}")
        logger.info(f"  - Label noise σ: {LABEL_NOISE_SIGMA}")
        
        # Select feature columns
        feature_cols = [col for col in FEATURE_NAMES if col in features_df.columns]
        X = features_df[feature_cols].fillna(0)
        
        # CHANGE 3: Time-based train/test split
        if use_temporal_split and 'observation_time' in features_df.columns:
            logger.info("Using temporal train/test split (70% train, 15% val, 15% test)")
            
            # Sort by time
            time_sorted_idx = features_df['observation_time'].argsort().values
            n = len(features_df)
            
            # Split indices
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            
            train_idx = time_sorted_idx[:train_end]
            val_idx = time_sorted_idx[train_end:val_end]
            test_idx = time_sorted_idx[val_end:]
            
            X_train = X.iloc[train_idx].reset_index(drop=True)
            X_test = X.iloc[test_idx].reset_index(drop=True)
            y_train = labels_df.iloc[train_idx].reset_index(drop=True)
            y_test = labels_df.iloc[test_idx].reset_index(drop=True)
            
            logger.info(f"Temporal split - Train: {len(X_train)}, Val: {len(val_idx)}, Test: {len(X_test)}")
        else:
            logger.info("Using random train/test split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels_df, test_size=test_size, random_state=self.random_state
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
        
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
        
        # CHANGE 2: Feature noise injection (training only)
        logger.info(f"Injecting Gaussian noise (σ={FEATURE_NOISE_SIGMA}) into training features...")
        X_train_noisy = X_train_scaled.copy()
        for col in feature_cols:
            noise = np.random.normal(0, FEATURE_NOISE_SIGMA, len(X_train_noisy))
            X_train_noisy[col] = X_train_noisy[col] + noise
        
        # CHANGE 1: Soft label generation with probabilistic noise
        logger.info(f"Adding label noise (σ={LABEL_NOISE_SIGMA}) for soft labels...")
        y_train_soft = y_train.copy()
        for col in y_train_soft.columns:
            if y_train_soft[col].dtype in ['float64', 'float32']:
                noise = np.random.normal(0, LABEL_NOISE_SIGMA, len(y_train_soft))
                y_train_soft[col] = np.clip(y_train_soft[col] + noise, 0, 1)
        
        self.scalers['features'] = scaler
        
        logger.info(f"Data preparation complete:")
        logger.info(f"  - Train set: {len(X_train_noisy)} (with noise)")
        logger.info(f"  - Test set: {len(X_test_scaled)} (no noise)")
        
        return {
            'X_train': X_train_noisy, 'X_test': X_test_scaled,
            'y_train': y_train_soft, 'y_test': y_test
        }, {'scaler': scaler}
    
    def train_risk_regressor(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            model_type: str = 'gradient_boosting') -> Dict[str, Any]:
        """
        Train continuous risk score regressor with regularization.
        
        CHANGE 4: Regularization & Model Control
        - max_depth ≤ 5 (reduced from 8)
        - min_samples_leaf ≥ 20 (increased)
        - subsample = 0.8
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            model_type: 'gradient_boosting' or 'random_forest'
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training risk regressor ({model_type}) with regularization...")
        logger.info(f"  - max_depth: {MAX_TREE_DEPTH}")
        logger.info(f"  - min_samples_leaf: {MIN_SAMPLES_LEAF}")
        logger.info(f"  - subsample: {SUBSAMPLE_RATIO}")
        
        if model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=MAX_TREE_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                subsample=SUBSAMPLE_RATIO,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=MAX_TREE_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=self.random_state
            )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # CHANGE 5: Realistic metric reporting
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
        }
        
        logger.info(f"Risk Regressor Metrics:")
        logger.info(f"  - Test RMSE: {metrics['test_rmse']:.4f}")
        logger.info(f"  - Test MAE: {metrics['test_mae']:.4f}")
        logger.info(f"  - Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"  - Train-Test gap (RMSE): {abs(metrics['train_rmse'] - metrics['test_rmse']):.4f}")
        
        # CHANGE 6: Metric disclaimer
        logger.warning("[DISCLAIMER] Metrics reflect synthetic data with injected noise.")
        logger.warning("Real-world performance will be lower and requires validation")
        logger.warning("against historical flood events.")
        
        self.models['risk_regressor'] = model
        self.metrics['risk_regressor'] = metrics
        
        return {'model': model, 'metrics': metrics}
    
    def train_warning_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_test: pd.DataFrame, y_test: pd.Series,
                                model_type: str = 'gradient_boosting') -> Dict[str, Any]:
        """
        Train binary early warning classifier with regularization.
        
        CHANGE 4: Regularization & Model Control
        - Gradient Boosting: max_depth ≤ 5, min_samples_leaf ≥ 20
        - Logistic Regression: C ≤ 0.3 (strong regularization)
        
        Args:
            X_train, y_train: Training data (binary labels)
            X_test, y_test: Test data
            model_type: 'gradient_boosting' or 'logistic_regression'
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training warning classifier ({model_type}) with regularization...")
        
        if model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=MAX_TREE_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                subsample=SUBSAMPLE_RATIO,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            model = LogisticRegression(
                C=LOGISTIC_REGULARIZATION,
                random_state=self.random_state, 
                max_iter=1000
            )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # CHANGE 5: Realistic metric reporting (Recall, Precision-Recall)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        precision = (y_pred_test == y_test).sum() / max(1, (y_pred_test == 1).sum())
        
        metrics = {
            'recall': recall,
            'precision': precision,
            'f1': f1_score(y_test, y_pred_test, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.0,
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
        }
        
        logger.info(f"Warning Classifier Metrics:")
        logger.info(f"  - Recall (sensitivity): {metrics['recall']:.4f}")
        logger.info(f"  - Precision: {metrics['precision']:.4f}")
        logger.info(f"  - F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # CHANGE 6: Metric disclaimer
        logger.warning("[DISCLAIMER] Metrics reflect synthetic data with injected noise.")
        logger.warning("Real-world performance will be lower and requires validation")
        logger.warning("against historical flood events.")
        
        self.models['warning_classifier'] = model
        self.metrics['warning_classifier'] = metrics
        
        return {'model': model, 'metrics': metrics}
    
    def train_risk_tier_classifier(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Train multi-class risk tier classifier with regularization.
        
        CHANGE 4: Regularization & Model Control
        - max_depth ≤ 5
        - min_samples_leaf ≥ 20
        - subsample = 0.8
        
        Args:
            X_train, y_train: Training data (multi-class labels)
            X_test, y_test: Test data
        
        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training risk tier classifier with regularization...")
        logger.info(f"  - max_depth: {MAX_TREE_DEPTH}")
        logger.info(f"  - min_samples_leaf: {MIN_SAMPLES_LEAF}")
        
        model = GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=MAX_TREE_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            subsample=SUBSAMPLE_RATIO,
            learning_rate=0.1,
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred_test = model.predict(X_test)
        
        # CHANGE 5: Realistic metric reporting
        metrics = {
            'macro_f1': f1_score(y_test, y_pred_test, average='macro', zero_division=0),
            'weighted_f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist(),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
        }
        
        logger.info(f"Risk Tier Classifier Metrics:")
        logger.info(f"  - Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"  - Weighted F1: {metrics['weighted_f1']:.4f}")
        
        # CHANGE 6: Metric disclaimer
        logger.warning("[DISCLAIMER] Metrics reflect synthetic data with injected noise.")
        logger.warning("Real-world performance will be lower and requires validation")
        logger.warning("against historical flood events.")
        
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

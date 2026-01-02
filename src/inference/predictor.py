"""
Flood Risk Predictor for Batch and Real-time Inference.

Data Source: Central Water Commission (CWC), Government of India

Generates predictions using trained models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from src.utils.logger import setup_logger
from src.utils.constants import FEATURE_NAMES, RISK_TIERS

logger = setup_logger(__name__)


class FloodPredictor:
    """
    Generates flood risk predictions using trained models.
    
    Provides:
    - Continuous risk scores
    - Binary early warning predictions
    - Multi-class risk tier classifications
    - Batch and real-time inference
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize predictor.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.load_models()
        logger.info("Flood predictor initialized")
    
    def load_models(self) -> None:
        """Load trained models and scalers from disk."""
        import os
        
        for model_name in ['risk_regressor', 'warning_classifier', 'risk_tier_classifier']:
            path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(path):
                try:
                    self.models[model_name] = joblib.load(path)
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {str(e)}")
        
        scaler_path = os.path.join(self.model_dir, "features_scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                self.scalers['features'] = joblib.load(scaler_path)
                logger.info("Loaded feature scaler")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {str(e)}")
    
    def predict_risk_score(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict continuous risk scores.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of risk scores (0-1)
        """
        if 'risk_regressor' not in self.models:
            logger.error("Risk regressor model not loaded")
            return np.array([])
        
        model = self.models['risk_regressor']
        
        # Prepare features
        X = self._prepare_features(features)
        
        if X is None or len(X) == 0:
            return np.array([])
        
        # Predict
        predictions = model.predict(X)
        
        # Clip to [0, 1] range
        predictions = np.clip(predictions, 0, 1)
        
        logger.info(f"Generated {len(predictions)} risk score predictions")
        
        return predictions
    
    def predict_warning(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binary early warning.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if 'warning_classifier' not in self.models:
            logger.error("Warning classifier model not loaded")
            return np.array([]), np.array([])
        
        model = self.models['warning_classifier']
        
        # Prepare features
        X = self._prepare_features(features)
        
        if X is None or len(X) == 0:
            return np.array([]), np.array([])
        
        # Predict
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        logger.info(f"Generated {len(predictions)} warning predictions")
        
        return predictions, probabilities
    
    def predict_risk_tier(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict multi-class risk tiers.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of risk tier predictions (0=low, 1=medium, 2=high)
        """
        if 'risk_tier_classifier' not in self.models:
            logger.error("Risk tier classifier model not loaded")
            return np.array([])
        
        model = self.models['risk_tier_classifier']
        
        # Prepare features
        X = self._prepare_features(features)
        
        if X is None or len(X) == 0:
            return np.array([])
        
        # Predict
        predictions = model.predict(X)
        
        logger.info(f"Generated {len(predictions)} risk tier predictions")
        
        return predictions
    
    def predict_all(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all predictions for features.
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            DataFrame with all predictions
        """
        logger.info(f"Generating predictions for {len(features)} records...")
        
        results = features[['station_id', 'observation_time']].copy() if 'station_id' in features.columns else pd.DataFrame()
        
        # Risk scores
        risk_scores = self.predict_risk_score(features)
        if len(risk_scores) > 0:
            results['risk_score'] = risk_scores
            results['risk_level'] = results['risk_score'].apply(self._score_to_level)
        
        # Warning predictions
        warnings, warning_probs = self.predict_warning(features)
        if len(warnings) > 0:
            results['warning_prediction'] = warnings
            results['warning_probability'] = warning_probs
        
        # Risk tiers
        risk_tiers = self.predict_risk_tier(features)
        if len(risk_tiers) > 0:
            results['risk_tier'] = risk_tiers
            results['risk_tier_name'] = results['risk_tier'].apply(self._tier_to_name)
        
        # Add feature columns for explanations
        feature_cols = ['distance_to_danger', 'rate_of_rise_3h', 'current_level', 'distance_to_warning']
        for col in feature_cols:
            if col in features.columns:
                results[col] = features[col].values
        
        logger.info(f"Predictions complete. Shape: {results.shape}")
        
        return results
    
    def predict_batch(self, features: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
        """
        Generate predictions in batches.
        
        Args:
            features: DataFrame with feature columns
            batch_size: Number of records per batch
            
        Returns:
            DataFrame with all predictions
        """
        logger.info(f"Processing {len(features)} records in batches of {batch_size}...")
        
        results = []
        
        for i in range(0, len(features), batch_size):
            batch = features.iloc[i:i+batch_size]
            batch_results = self.predict_all(batch)
            results.append(batch_results)
        
        final_results = pd.concat(results, ignore_index=True)
        
        logger.info(f"Batch processing complete. Total predictions: {len(final_results)}")
        
        return final_results
    
    def _prepare_features(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Prepare features for model input.
        
        Args:
            features: Raw feature DataFrame
            
        Returns:
            Scaled feature DataFrame or None if error
        """
        try:
            # Select feature columns
            feature_cols = [col for col in FEATURE_NAMES if col in features.columns]
            
            if not feature_cols:
                logger.error("No valid feature columns found")
                return None
            
            X = features[feature_cols].fillna(0)
            
            # Scale features if scaler available
            if 'features' in self.scalers:
                X_scaled = self.scalers['features'].transform(X)
                X = pd.DataFrame(X_scaled, columns=feature_cols)
            
            return X
        
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
    
    def _score_to_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score < RISK_TIERS['low'][1]:
            return 'low'
        elif score < RISK_TIERS['medium'][1]:
            return 'medium'
        else:
            return 'high'
    
    def _tier_to_name(self, tier: int) -> str:
        """Convert tier index to name."""
        tier_names = {0: 'low', 1: 'medium', 2: 'high'}
        return tier_names.get(tier, 'unknown')
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of loaded models."""
        return {
            'risk_regressor': 'risk_regressor' in self.models,
            'warning_classifier': 'warning_classifier' in self.models,
            'risk_tier_classifier': 'risk_tier_classifier' in self.models,
            'scaler': 'features' in self.scalers
        }

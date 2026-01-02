"""
Weather-Aware Flood Risk Predictor.

Extends base predictor with:
- Dual-mode inference (hydro-only vs weather-aware)
- Multi-horizon predictions (now, +6h, +24h)
- Weather-based risk explanations
- Backward compatibility with existing models

BACKWARD-COMPATIBLE:
- Existing models work unchanged
- Weather features are optional
- Falls back to hydro-only if weather unavailable
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from src.inference.predictor import FloodPredictor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class WeatherAwarePredictor(FloodPredictor):
    """
    Weather-aware flood risk predictor.
    
    Extends base predictor with:
    - Multi-horizon predictions (0h, +6h, +24h)
    - Weather-based risk factors
    - Dual-mode inference
    """
    
    def __init__(self, model_dir: str = "data/models", use_weather_models: bool = False):
        """
        Initialize weather-aware predictor.
        
        Args:
            model_dir: Directory containing trained models
            use_weather_models: Whether to use weather-aware models (if available)
        """
        super().__init__(model_dir)
        self.use_weather_models = use_weather_models
        self.weather_models = {}
        
        if use_weather_models:
            self._load_weather_models()
        
        logger.info(f"Weather-aware predictor initialized (weather_models={use_weather_models})")
    
    def predict_with_weather(self, features: pd.DataFrame,
                            weather_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate predictions with optional weather data.
        
        Args:
            features: Feature DataFrame (with weather features if available)
            weather_data: Optional weather data for context
            
        Returns:
            DataFrame with predictions and weather context
        """
        logger.info(f"Generating weather-aware predictions for {len(features)} stations...")
        
        # Determine mode
        weather_available = weather_data is not None and not weather_data.empty
        mode = "WEATHER-AWARE" if weather_available else "HYDRO-ONLY"
        logger.info(f"Prediction mode: {mode}")
        
        # Generate base predictions
        predictions = self.predict_all(features)
        
        # Add weather context
        if weather_available:
            predictions = self._add_weather_context(predictions, weather_data)
        
        # Add prediction horizons
        predictions = self._add_prediction_horizons(predictions, features)
        
        # Add mode indicator
        predictions['prediction_mode'] = mode
        predictions['weather_data_used'] = weather_available
        
        logger.info(f"Generated {len(predictions)} predictions in {mode} mode")
        
        return predictions
    
    def _add_weather_context(self, predictions: pd.DataFrame,
                            weather_data: pd.DataFrame) -> pd.DataFrame:
        """
        Add weather context to predictions.
        
        Args:
            predictions: Predictions DataFrame
            weather_data: Weather data DataFrame
            
        Returns:
            Predictions with weather context
        """
        # Merge weather data
        predictions = predictions.merge(
            weather_data[['station_id', 'rainfall_last_6h_mm',
                         'rainfall_forecast_next_6h_mm',
                         'rainfall_forecast_next_24h_mm']],
            on='station_id',
            how='left'
        )
        
        # Fill missing weather data with 0
        for col in ['rainfall_last_6h_mm', 'rainfall_forecast_next_6h_mm',
                   'rainfall_forecast_next_24h_mm']:
            if col in predictions.columns:
                predictions[col] = predictions[col].fillna(0.0)
        
        return predictions
    
    def _add_prediction_horizons(self, predictions: pd.DataFrame,
                                features: pd.DataFrame) -> pd.DataFrame:
        """
        Add multi-horizon predictions (now, +6h, +24h).
        
        Args:
            predictions: Current predictions
            features: Feature DataFrame
            
        Returns:
            Predictions with horizon information
        """
        # Current prediction (now)
        predictions['risk_score_now'] = predictions['risk_score']
        predictions['risk_tier_now'] = predictions['risk_tier_name']
        
        # +6 hour prediction (based on rainfall forecast)
        if 'rainfall_forecast_next_6h_mm' in features.columns:
            # Adjust risk based on forecast rainfall
            rainfall_6h = features.get('rainfall_forecast_next_6h_mm', 0)
            rainfall_adjustment = np.minimum(rainfall_6h / 50.0, 0.3)  # Max +30% adjustment
            predictions['risk_score_6h'] = np.clip(
                predictions['risk_score'] + rainfall_adjustment, 0, 1
            )
        else:
            predictions['risk_score_6h'] = predictions['risk_score']
        
        # +24 hour prediction (based on 24h rainfall forecast)
        if 'rainfall_forecast_next_24h_mm' in features.columns:
            # Adjust risk based on forecast rainfall
            rainfall_24h = features.get('rainfall_forecast_next_24h_mm', 0)
            rainfall_adjustment = np.minimum(rainfall_24h / 100.0, 0.5)  # Max +50% adjustment
            predictions['risk_score_24h'] = np.clip(
                predictions['risk_score'] + rainfall_adjustment, 0, 1
            )
        else:
            predictions['risk_score_24h'] = predictions['risk_score']
        
        # Determine risk tiers for future horizons
        predictions['risk_tier_6h'] = predictions['risk_score_6h'].apply(
            self._score_to_tier
        )
        predictions['risk_tier_24h'] = predictions['risk_score_24h'].apply(
            self._score_to_tier
        )
        
        return predictions
    
    def _score_to_tier(self, score: float) -> str:
        """
        Convert risk score to tier name.
        
        Args:
            score: Risk score (0-1)
            
        Returns:
            Risk tier name (LOW/MEDIUM/HIGH)
        """
        if score < 0.33:
            return 'LOW'
        elif score < 0.67:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def _load_weather_models(self) -> None:
        """Load weather-aware models if available."""
        import os
        import joblib
        
        weather_model_names = [
            'risk_regressor_weather',
            'warning_classifier_weather',
            'risk_tier_classifier_weather'
        ]
        
        for model_name in weather_model_names:
            path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(path):
                try:
                    self.weather_models[model_name] = joblib.load(path)
                    logger.info(f"Loaded weather model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {str(e)}")
    
    def get_prediction_status(self) -> Dict[str, any]:
        """
        Get status of prediction system.
        
        Returns:
            Dictionary with prediction status
        """
        return {
            'base_models_loaded': len(self.models) > 0,
            'weather_models_loaded': len(self.weather_models) > 0,
            'use_weather_models': self.use_weather_models,
            'models': list(self.models.keys()),
            'weather_models': list(self.weather_models.keys()),
        }

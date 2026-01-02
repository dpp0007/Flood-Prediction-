"""
SHAP-based Model Explanations for Flood Predictions.

Data Source: Central Water Commission (CWC), Government of India

Generates human-readable explanations for model predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import joblib
from src.utils.logger import setup_logger
from src.utils.constants import TOP_FEATURES_FOR_EXPLANATION, FEATURE_NAMES, RISK_TIERS

logger = setup_logger(__name__)


class ExplanationGenerator:
    """
    Generates explanations for flood risk predictions.
    
    Provides:
    - Feature contribution analysis
    - Risk factor identification
    - Station-specific risk drivers
    - Alert justification text
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize explanation generator.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        logger.info("Explanation generator initialized")
    
    def load_models(self) -> None:
        """Load trained models and scalers from disk."""
        import os
        
        for model_name in ['risk_regressor', 'warning_classifier', 'risk_tier_classifier']:
            path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(path):
                self.models[model_name] = joblib.load(path)
                logger.info(f"Loaded model: {model_name}")
        
        scaler_path = os.path.join(self.model_dir, "features_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scalers['features'] = joblib.load(scaler_path)
            logger.info("Loaded feature scaler")
    
    def explain_risk_score(self, features: pd.Series, risk_score: float,
                          top_n: int = None) -> Dict[str, any]:
        """
        Explain a continuous risk score prediction.
        
        Args:
            features: Feature values for the prediction
            risk_score: Predicted risk score (0-1)
            top_n: Number of top contributing features
            
        Returns:
            Dictionary with explanation components
        """
        if top_n is None:
            top_n = TOP_FEATURES_FOR_EXPLANATION
        
        if 'risk_regressor' not in self.models:
            logger.warning("Risk regressor model not loaded")
            return {}
        
        model = self.models['risk_regressor']
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Calculate feature contributions (simplified)
        feature_values = features[FEATURE_NAMES[:len(importances)]].values
        contributions = feature_values * importances
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
        
        explanation = {
            'risk_score': round(risk_score, 3),
            'risk_level': self._get_risk_level(risk_score),
            'top_factors': [
                {
                    'feature': FEATURE_NAMES[i],
                    'value': round(feature_values[i], 3),
                    'contribution': round(contributions[i], 3),
                    'importance': round(importances[i], 3)
                }
                for i in top_indices
            ],
            'summary': self._generate_risk_summary(risk_score, features)
        }
        
        return explanation
    
    def explain_warning_prediction(self, features: pd.Series, warning_prob: float,
                                   top_n: int = None) -> Dict[str, any]:
        """
        Explain a binary early warning prediction.
        
        Args:
            features: Feature values for the prediction
            warning_prob: Predicted warning probability (0-1)
            top_n: Number of top contributing features
            
        Returns:
            Dictionary with explanation components
        """
        if top_n is None:
            top_n = TOP_FEATURES_FOR_EXPLANATION
        
        if 'warning_classifier' not in self.models:
            logger.warning("Warning classifier model not loaded")
            return {}
        
        model = self.models['warning_classifier']
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Calculate feature contributions
        feature_values = features[FEATURE_NAMES[:len(importances)]].values
        contributions = feature_values * importances
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
        
        explanation = {
            'warning_probability': round(warning_prob, 3),
            'warning_triggered': warning_prob > 0.5,
            'top_factors': [
                {
                    'feature': FEATURE_NAMES[i],
                    'value': round(feature_values[i], 3),
                    'contribution': round(contributions[i], 3),
                    'importance': round(importances[i], 3)
                }
                for i in top_indices
            ],
            'summary': self._generate_warning_summary(warning_prob, features)
        }
        
        return explanation
    
    def explain_risk_tier(self, features: pd.Series, risk_tier: str,
                         top_n: int = None) -> Dict[str, any]:
        """
        Explain a multi-class risk tier prediction.
        
        Args:
            features: Feature values for the prediction
            risk_tier: Predicted risk tier (low/medium/high)
            top_n: Number of top contributing features
            
        Returns:
            Dictionary with explanation components
        """
        if top_n is None:
            top_n = TOP_FEATURES_FOR_EXPLANATION
        
        if 'risk_tier_classifier' not in self.models:
            logger.warning("Risk tier classifier model not loaded")
            return {}
        
        model = self.models['risk_tier_classifier']
        
        # Get feature importance
        importances = model.feature_importances_
        
        # Calculate feature contributions
        feature_values = features[FEATURE_NAMES[:len(importances)]].values
        contributions = feature_values * importances
        
        # Get top contributing features
        top_indices = np.argsort(np.abs(contributions))[-top_n:][::-1]
        
        explanation = {
            'risk_tier': risk_tier,
            'top_factors': [
                {
                    'feature': FEATURE_NAMES[i],
                    'value': round(feature_values[i], 3),
                    'contribution': round(contributions[i], 3),
                    'importance': round(importances[i], 3)
                }
                for i in top_indices
            ],
            'summary': self._generate_tier_summary(risk_tier, features)
        }
        
        return explanation
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level."""
        if risk_score < RISK_TIERS['low'][1]:
            return 'low'
        elif risk_score < RISK_TIERS['medium'][1]:
            return 'medium'
        else:
            return 'high'
    
    def _generate_risk_summary(self, risk_score: float, features: pd.Series) -> str:
        """Generate human-readable risk summary."""
        risk_level = self._get_risk_level(risk_score)
        
        distance_to_danger = features.get('distance_to_danger', 0)
        rate_of_rise = features.get('rate_of_rise_3h', 0)
        
        summary = f"Risk Score: {risk_score:.1%} ({risk_level.upper()}). "
        
        if distance_to_danger < 0.5:
            summary += "Water level is dangerously close to danger threshold. "
        elif distance_to_danger < 1.0:
            summary += "Water level is approaching danger threshold. "
        
        if rate_of_rise > 0.1:
            summary += f"Water rising at {rate_of_rise:.2f}m/hour. "
        
        return summary
    
    def _generate_warning_summary(self, warning_prob: float, features: pd.Series) -> str:
        """Generate human-readable warning summary."""
        summary = f"Warning Probability: {warning_prob:.1%}. "
        
        if warning_prob > 0.7:
            summary += "HIGH likelihood of flood warning in next 6 hours. "
        elif warning_prob > 0.5:
            summary += "MODERATE likelihood of flood warning in next 6 hours. "
        else:
            summary += "LOW likelihood of flood warning in next 6 hours. "
        
        rate_of_rise = features.get('rate_of_rise_3h', 0)
        if rate_of_rise > 0.15:
            summary += "Rapid water level rise detected. "
        
        return summary
    
    def _generate_tier_summary(self, risk_tier: str, features: pd.Series) -> str:
        """Generate human-readable tier summary."""
        tier_descriptions = {
            'low': 'Low flood risk. Normal conditions.',
            'medium': 'Moderate flood risk. Monitor water levels closely.',
            'high': 'High flood risk. Prepare for potential evacuation.'
        }
        
        summary = tier_descriptions.get(risk_tier, 'Unknown risk tier.')
        
        distance_to_warning = features.get('distance_to_warning', 0)
        if distance_to_warning < 0.5:
            summary += " Water level near warning threshold."
        
        return summary
    
    def generate_alert_text(self, station_id: str, station_name: str,
                           risk_score: float, warning_prob: float,
                           risk_tier: str, features: pd.Series) -> str:
        """
        Generate complete alert text for a station.
        
        Args:
            station_id: Station identifier
            station_name: Station name
            risk_score: Continuous risk score
            warning_prob: Warning probability
            risk_tier: Risk tier classification
            features: Feature values
            
        Returns:
            Formatted alert text
        """
        alert = f"""
FLOOD RISK ALERT - {station_name} (ID: {station_id})
{'='*60}

RISK ASSESSMENT:
- Risk Score: {risk_score:.1%} ({self._get_risk_level(risk_score).upper()})
- Risk Tier: {risk_tier.upper()}
- Warning Probability: {warning_prob:.1%}

CURRENT CONDITIONS:
- Water Level: {features.get('current_level', 'N/A'):.2f}m
- Distance to Warning: {features.get('distance_to_warning', 'N/A'):.2f}m
- Distance to Danger: {features.get('distance_to_danger', 'N/A'):.2f}m
- Rate of Rise (3h): {features.get('rate_of_rise_3h', 'N/A'):.3f}m/h
- Trend: {'RISING' if features.get('rate_of_rise_3h', 0) > 0 else 'FALLING'}

KEY RISK FACTORS:
{self._format_risk_factors(features)}

RECOMMENDATION:
{self._get_recommendation(risk_tier, warning_prob)}

Data Source: Central Water Commission (CWC), Government of India
Disclaimer: This alert is for research/demonstration purposes only.
Does NOT replace official CWC flood warnings.
{'='*60}
"""
        return alert
    
    def _format_risk_factors(self, features: pd.Series) -> str:
        """Format risk factors for display."""
        factors = []
        
        if features.get('rate_of_rise_3h', 0) > 0.1:
            factors.append(f"  • Rapid water rise: {features['rate_of_rise_3h']:.3f}m/h")
        
        if features.get('distance_to_danger', 0) < 1.0:
            factors.append(f"  • Close to danger level: {features['distance_to_danger']:.2f}m")
        
        if features.get('consecutive_rising_hours', 0) > 6:
            factors.append(f"  • Sustained rise: {features['consecutive_rising_hours']:.0f} hours")
        
        if features.get('historical_percentile', 0) > 80:
            factors.append(f"  • High historical level: {features['historical_percentile']:.0f}th percentile")
        
        return '\n'.join(factors) if factors else "  • No critical factors identified"
    
    def _get_recommendation(self, risk_tier: str, warning_prob: float) -> str:
        """Get recommendation based on risk assessment."""
        if risk_tier == 'high' or warning_prob > 0.7:
            return "URGENT: Prepare for potential evacuation. Contact local authorities."
        elif risk_tier == 'medium' or warning_prob > 0.5:
            return "CAUTION: Monitor situation closely. Be prepared to evacuate if conditions worsen."
        else:
            return "Continue normal monitoring. No immediate action required."

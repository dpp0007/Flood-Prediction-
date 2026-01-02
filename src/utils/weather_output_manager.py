"""
Weather-Aware Output Manager Extension.

Extends OutputManager to include weather data in JSON predictions.
Generates JSON in GFS-aware format with forecast signals and risk adjustments.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class WeatherAwareOutputManager:
    """
    Extends output management with weather-aware features.
    
    Generates JSON with:
    - forecast_signal: GFS weather data
    - risk_adjustment: How GFS affects risk
    - confidence: Combined model + forecast confidence
    - time_context: Forecast horizon and windows
    """
    
    @staticmethod
    def build_gfs_aware_json(row: pd.Series, location_data: Dict, station_id: str) -> Dict[str, Any]:
        """
        Build GFS-aware JSON prediction with weather signals.
        
        Args:
            row: Prediction row with weather data
            location_data: Location context
            station_id: Station identifier
            
        Returns:
            GFS-aware prediction JSON
        """
        risk_score = float(row.get('risk_score', 0.0))
        risk_tier = str(row.get('risk_tier_name', 'unknown')).lower()
        
        # Determine risk level
        if risk_tier == 'high':
            risk_level = 'HIGH'
        elif risk_tier == 'medium':
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Get weather data
        rainfall_last_6h = float(row.get('rainfall_last_6h_mm', 0))
        rainfall_forecast_6h = float(row.get('rainfall_forecast_next_6h_mm', 0))
        rainfall_forecast_24h = float(row.get('rainfall_forecast_next_24h_mm', 0))
        
        # Determine rainfall trend
        if rainfall_forecast_6h > rainfall_last_6h * 1.5:
            rainfall_trend = "INCREASING"
        elif rainfall_forecast_6h < rainfall_last_6h * 0.5:
            rainfall_trend = "DECREASING"
        else:
            rainfall_trend = "STABLE"
        
        # Calculate extreme rain probability (0-1)
        extreme_rain_probability = min(rainfall_forecast_24h / 100.0, 1.0)
        
        # Calculate baseline risk (without weather)
        baseline_risk_score = risk_score * 0.7  # Reduce by weather contribution
        if baseline_risk_score < 0.33:
            baseline_risk = "LOW"
        elif baseline_risk_score < 0.67:
            baseline_risk = "MEDIUM"
        else:
            baseline_risk = "HIGH"
        
        # Calculate adjustment reason
        if rainfall_forecast_24h > 50:
            adjustment_reason = "Heavy rainfall forecasted by GFS"
        elif rainfall_forecast_24h > 25:
            adjustment_reason = "Moderate rainfall forecasted by GFS"
        elif rainfall_forecast_6h > 10:
            adjustment_reason = "Rainfall expected in next 6 hours"
        else:
            adjustment_reason = "Stable conditions forecasted"
        
        # Calculate confidence scores
        model_confidence = 0.75 + (risk_score * 0.15)  # Higher risk = higher confidence
        forecast_support = min(extreme_rain_probability, 0.95)
        combined_confidence = (model_confidence + forecast_support) / 2.0
        
        # Build JSON structure (Coordination Agent schema compliant)
        prediction_time = datetime.utcnow()
        valid_until = prediction_time + timedelta(hours=24)
        
        prediction_json = {
            "prediction": {
                "flood_probability": round(risk_score, 4),
                "risk_level": risk_level,
            },
            "confidence": {
                "model_confidence": round(model_confidence, 4),
                "forecast_support": round(forecast_support, 4),
                "combined_confidence": round(combined_confidence, 4),
            },
            "forecast_signal": {
                "source": "GFS",
                "forecast_horizon_hours": 24,
                "rainfall_trend": rainfall_trend,
            },
            "context": {
                "location": location_data or {
                    "district": "Unknown",
                    "state": "Unknown",
                    "latitude": 0.0,
                    "longitude": 0.0,
                },
                "time_window": {
                    "prediction_time": prediction_time.isoformat() + "Z",
                    "valid_until": valid_until.isoformat() + "Z",
                }
            },
            "explanation": [],
            "metadata": {
                "model_type": "hybrid",
                "model_version": "1.1",
                "data_sources": ["Central Water Commission (CWC), Government of India", "NOAA GFS"],
            }
        }
        
        return prediction_json
    
    @staticmethod
    def extend_prediction_json_with_weather(prediction_json: Dict[str, Any],
                                           row: pd.Series) -> Dict[str, Any]:
        """
        Extend prediction JSON with weather data.
        
        BACKWARD-COMPATIBLE: Only adds to metadata, doesn't change existing structure.
        
        Args:
            prediction_json: Existing prediction JSON
            row: Prediction row with weather data
            
        Returns:
            Extended prediction JSON
        """
        # Check if weather data is available
        weather_available = (
            'rainfall_forecast_next_6h_mm' in row and 
            row['rainfall_forecast_next_6h_mm'] is not None and
            row['rainfall_forecast_next_6h_mm'] > 0
        )
        
        # Add weather metadata
        if 'metadata' not in prediction_json:
            prediction_json['metadata'] = {}
        
        prediction_json['metadata']['weather_data_used'] = weather_available
        prediction_json['metadata']['weather_source'] = ['GFS']
        prediction_json['metadata']['prediction_horizon_hours'] = [0, 6, 24]
        
        # Add weather-specific fields if available
        if weather_available:
            # Add rainfall forecasts
            prediction_json['metadata']['rainfall_forecast'] = {
                'last_6h_mm': round(float(row.get('rainfall_last_6h_mm', 0)), 2),
                'next_6h_mm': round(float(row.get('rainfall_forecast_next_6h_mm', 0)), 2),
                'next_24h_mm': round(float(row.get('rainfall_forecast_next_24h_mm', 0)), 2),
            }
            
            # Add multi-horizon risk scores if available
            if 'risk_score_6h' in row:
                prediction_json['prediction']['risk_score_6h'] = round(float(row['risk_score_6h']), 4)
                prediction_json['prediction']['risk_level_6h'] = row.get('risk_tier_6h', 'UNKNOWN')
            
            if 'risk_score_24h' in row:
                prediction_json['prediction']['risk_score_24h'] = round(float(row['risk_score_24h']), 4)
                prediction_json['prediction']['risk_level_24h'] = row.get('risk_tier_24h', 'UNKNOWN')
        
        return prediction_json
    
    @staticmethod
    def generate_weather_explanation(row: pd.Series, risk_level: str) -> str:
        """
        Generate weather-based explanation for prediction.
        
        Args:
            row: Prediction row
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            
        Returns:
            Weather-based explanation string
        """
        explanations = []
        
        # Check if weather data is available
        rainfall_6h = float(row.get('rainfall_forecast_next_6h_mm', 0))
        rainfall_24h = float(row.get('rainfall_forecast_next_24h_mm', 0))
        
        if rainfall_6h > 0 or rainfall_24h > 0:
            # Add weather-based explanation
            if rainfall_24h > 50:
                explanations.append(
                    f"Heavy rainfall forecasted in next 24 hours ({rainfall_24h:.1f}mm). "
                    f"Significant flood risk expected."
                )
            elif rainfall_24h > 25:
                explanations.append(
                    f"Moderate rainfall forecasted in next 24 hours ({rainfall_24h:.1f}mm). "
                    f"Monitor water levels closely."
                )
            elif rainfall_6h > 10:
                explanations.append(
                    f"Rainfall expected in next 6 hours ({rainfall_6h:.1f}mm). "
                    f"Water levels may rise."
                )
            
            # Add trend information
            rainfall_last_6h = float(row.get('rainfall_last_6h_mm', 0))
            if rainfall_6h > rainfall_last_6h:
                explanations.append(
                    f"Rainfall intensity increasing (from {rainfall_last_6h:.1f}mm to {rainfall_6h:.1f}mm). "
                    f"Risk may escalate."
                )
        
        return " ".join(explanations) if explanations else ""
    
    @staticmethod
    def add_predictive_alert_flag(prediction_json: Dict[str, Any],
                                 row: pd.Series) -> Dict[str, Any]:
        """
        Add predictive alert flag if future risk is HIGH.
        
        Args:
            prediction_json: Prediction JSON
            row: Prediction row
            
        Returns:
            Prediction JSON with alert flag
        """
        # Check if future risk is HIGH
        risk_6h = row.get('risk_tier_6h', 'LOW')
        risk_24h = row.get('risk_tier_24h', 'LOW')
        
        if risk_6h == 'HIGH' or risk_24h == 'HIGH':
            if 'alerts' not in prediction_json:
                prediction_json['alerts'] = []
            
            prediction_json['alerts'].append({
                'type': 'PREDICTIVE_FLOOD_ALERT',
                'severity': 'HIGH',
                'description': 'Forecast-based alert (weather-aware)',
                'horizon_hours': 6 if risk_6h == 'HIGH' else 24,
                'timestamp': datetime.utcnow().isoformat() + 'Z',
            })
        
        return prediction_json

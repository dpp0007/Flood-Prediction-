"""
Inference Module

Data Source: Central Water Commission (CWC), Government of India
Website: https://inf.cwc.gov.in/

Provides batch and real-time prediction capabilities.
"""

from src.inference.predictor import FloodPredictor
from src.inference.alert_generator import AlertGenerator

__all__ = ["FloodPredictor", "AlertGenerator"]

"""
Feature Importance Analysis for Flood Prediction Models.

Data Source: Central Water Commission (CWC), Government of India

Analyzes which features are most important for model predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import joblib
from src.utils.logger import setup_logger
from src.utils.constants import TOP_FEATURES_FOR_EXPLANATION, FEATURE_NAMES

logger = setup_logger(__name__)


class FeatureImportance:
    """
    Analyzes feature importance for trained models.
    
    Supports:
    - Tree-based model feature importance
    - Permutation importance
    - Station-wise feature rankings
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize feature importance analyzer.
        
        Args:
            model_dir: Directory containing trained models
        """
        self.model_dir = model_dir
        self.models = {}
        self.feature_importances = {}
        logger.info("Feature importance analyzer initialized")
    
    def load_models(self) -> None:
        """Load trained models from disk."""
        import os
        
        for model_name in ['risk_regressor', 'warning_classifier', 'risk_tier_classifier']:
            path = os.path.join(self.model_dir, f"{model_name}.pkl")
            if os.path.exists(path):
                self.models[model_name] = joblib.load(path)
                logger.info(f"Loaded model: {model_name}")
    
    def get_feature_importance(self, model_name: str, top_n: int = None) -> pd.DataFrame:
        """
        Get feature importance for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance scores
        """
        if top_n is None:
            top_n = TOP_FEATURES_FOR_EXPLANATION
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not loaded")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        importances = model.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': FEATURE_NAMES[:len(importances)],
            'importance': importances,
            'importance_pct': (importances / importances.sum() * 100).round(2)
        }).sort_values('importance', ascending=False).head(top_n)
        
        logger.info(f"Top {top_n} features for {model_name}:\n{importance_df}")
        
        return importance_df
    
    def get_all_importances(self, top_n: int = None) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance for all models.
        
        Args:
            top_n: Number of top features per model
            
        Returns:
            Dictionary mapping model names to importance DataFrames
        """
        importances = {}
        
        for model_name in self.models.keys():
            importances[model_name] = self.get_feature_importance(model_name, top_n)
        
        return importances
    
    def get_station_feature_importance(self, model_name: str, station_id: str,
                                       features: pd.DataFrame) -> pd.DataFrame:
        """
        Get feature importance for a specific station.
        
        Args:
            model_name: Name of the model
            station_id: Station identifier
            features: Feature DataFrame for the station
            
        Returns:
            DataFrame with feature values and importance
        """
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not loaded")
            return pd.DataFrame()
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        # Get station features
        station_features = features[features['station_id'] == station_id]
        
        if station_features.empty:
            logger.warning(f"No features found for station {station_id}")
            return pd.DataFrame()
        
        importances = model.feature_importances_
        
        # Create station-specific importance dataframe
        station_importance = pd.DataFrame({
            'feature': FEATURE_NAMES[:len(importances)],
            'importance': importances,
            'value': station_features[FEATURE_NAMES[:len(importances)]].iloc[0].values
        }).sort_values('importance', ascending=False)
        
        return station_importance
    
    def plot_feature_importance(self, model_name: str, top_n: int = None,
                               save_path: Optional[str] = None) -> None:
        """
        Plot feature importance.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to plot
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
            return
        
        importance_df = self.get_feature_importance(model_name, top_n)
        
        if importance_df.empty:
            logger.warning(f"No importance data for {model_name}")
            return
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance_pct'])
        plt.xlabel('Importance (%)')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()

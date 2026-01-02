"""
Alert Generation from Flood Predictions.

Data Source: Central Water Commission (CWC), Government of India

Generates actionable alerts from model predictions.
"""

import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from src.utils.logger import setup_logger
from src.explainability.explanations import ExplanationGenerator

logger = setup_logger(__name__)


class AlertGenerator:
    """
    Generates flood alerts from predictions.
    
    Provides:
    - Alert severity classification
    - Alert text generation
    - Alert filtering and prioritization
    - Alert history tracking
    """
    
    def __init__(self, explanation_generator: Optional[ExplanationGenerator] = None):
        """
        Initialize alert generator.
        
        Args:
            explanation_generator: Optional explanation generator for detailed alerts
        """
        self.explanation_generator = explanation_generator or ExplanationGenerator()
        self.alert_history = []
        logger.info("Alert generator initialized")
    
    def generate_alerts(self, predictions: pd.DataFrame,
                       station_metadata: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate alerts from predictions.
        
        Args:
            predictions: DataFrame with predictions
            station_metadata: Optional DataFrame with station information
            
        Returns:
            DataFrame with generated alerts
        """
        logger.info(f"Generating alerts for {len(predictions)} predictions...")
        
        alerts = predictions.copy()
        
        # Determine alert severity
        alerts['alert_severity'] = alerts.apply(
            lambda row: self._determine_severity(row),
            axis=1
        )
        
        # Determine if alert should be triggered
        alerts['alert_triggered'] = alerts['alert_severity'].isin(['MEDIUM', 'HIGH', 'CRITICAL'])
        
        # Generate alert messages
        alerts['alert_message'] = alerts.apply(
            lambda row: self._generate_alert_message(row),
            axis=1
        )
        
        # Add timestamp
        alerts['alert_timestamp'] = datetime.utcnow().isoformat()
        
        # Add station metadata if provided
        if station_metadata is not None:
            alerts = alerts.merge(
                station_metadata[['station_id', 'station_name', 'river_name', 'state']],
                on='station_id',
                how='left'
            )
        
        # Filter to triggered alerts
        triggered_alerts = alerts[alerts['alert_triggered']].copy()
        
        logger.info(f"Generated {len(triggered_alerts)} triggered alerts")
        
        # Track alert history
        self.alert_history.extend(triggered_alerts.to_dict('records'))
        
        return triggered_alerts
    
    def _determine_severity(self, row: pd.Series) -> str:
        """
        Determine alert severity.
        
        Args:
            row: Prediction row
            
        Returns:
            Severity level: NONE, LOW, MEDIUM, HIGH, CRITICAL
        """
        risk_score = row.get('risk_score', 0)
        warning_prob = row.get('warning_probability', 0)
        risk_tier = row.get('risk_tier_name', 'low')
        
        # Critical: High risk score + high warning probability
        if risk_score > 0.8 and warning_prob > 0.7:
            return 'CRITICAL'
        
        # High: High risk tier or high warning probability
        if risk_tier == 'high' or warning_prob > 0.7:
            return 'HIGH'
        
        # Medium: Medium risk tier or moderate warning probability
        if risk_tier == 'medium' or warning_prob > 0.5:
            return 'MEDIUM'
        
        # Low: Low risk tier but some warning signal
        if warning_prob > 0.3:
            return 'LOW'
        
        return 'NONE'
    
    def _generate_alert_message(self, row: pd.Series) -> str:
        """
        Generate alert message.
        
        Args:
            row: Prediction row
            
        Returns:
            Alert message string
        """
        severity = row.get('alert_severity', 'NONE')
        station_name = row.get('station_name', 'Unknown Station')
        risk_score = row.get('risk_score', 0)
        warning_prob = row.get('warning_probability', 0)
        
        if severity == 'CRITICAL':
            return (f"🚨 CRITICAL FLOOD ALERT - {station_name}: "
                   f"Risk {risk_score:.0%}, Warning Probability {warning_prob:.0%}. "
                   f"IMMEDIATE ACTION REQUIRED.")
        
        elif severity == 'HIGH':
            return (f"⚠️ HIGH FLOOD ALERT - {station_name}: "
                   f"Risk {risk_score:.0%}, Warning Probability {warning_prob:.0%}. "
                   f"Prepare for potential evacuation.")
        
        elif severity == 'MEDIUM':
            return (f"⚡ MODERATE FLOOD ALERT - {station_name}: "
                   f"Risk {risk_score:.0%}, Warning Probability {warning_prob:.0%}. "
                   f"Monitor situation closely.")
        
        elif severity == 'LOW':
            return (f"ℹ️ LOW FLOOD ALERT - {station_name}: "
                   f"Risk {risk_score:.0%}, Warning Probability {warning_prob:.0%}. "
                   f"Continue monitoring.")
        
        else:
            return f"✓ No alert - {station_name}: Risk {risk_score:.0%}"
    
    def filter_alerts_by_severity(self, alerts: pd.DataFrame,
                                 min_severity: str = 'MEDIUM') -> pd.DataFrame:
        """
        Filter alerts by minimum severity.
        
        Args:
            alerts: DataFrame with alerts
            min_severity: Minimum severity level
            
        Returns:
            Filtered alerts DataFrame
        """
        severity_order = {'NONE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        min_level = severity_order.get(min_severity, 0)
        
        filtered = alerts[
            alerts['alert_severity'].map(severity_order).fillna(0) >= min_level
        ]
        
        logger.info(f"Filtered to {len(filtered)} alerts with severity >= {min_severity}")
        
        return filtered
    
    def filter_alerts_by_region(self, alerts: pd.DataFrame,
                               states: List[str]) -> pd.DataFrame:
        """
        Filter alerts by region/state.
        
        Args:
            alerts: DataFrame with alerts
            states: List of state names
            
        Returns:
            Filtered alerts DataFrame
        """
        filtered = alerts[alerts['state'].isin(states)]
        
        logger.info(f"Filtered to {len(filtered)} alerts in states: {states}")
        
        return filtered
    
    def prioritize_alerts(self, alerts: pd.DataFrame,
                         top_n: int = 10) -> pd.DataFrame:
        """
        Prioritize alerts by severity and risk score.
        
        Args:
            alerts: DataFrame with alerts
            top_n: Number of top alerts to return
            
        Returns:
            Top N prioritized alerts
        """
        severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'NONE': 0}
        
        alerts['severity_rank'] = alerts['alert_severity'].map(severity_order)
        
        prioritized = alerts.sort_values(
            ['severity_rank', 'risk_score'],
            ascending=[False, False]
        ).head(top_n)
        
        logger.info(f"Prioritized to top {len(prioritized)} alerts")
        
        return prioritized
    
    def export_alerts(self, alerts: pd.DataFrame, output_path: str,
                     format: str = 'csv') -> None:
        """
        Export alerts to file.
        
        Args:
            alerts: DataFrame with alerts
            output_path: Path to save alerts
            format: Export format (csv, json, html)
        """
        try:
            if format == 'csv':
                alerts.to_csv(output_path, index=False)
            elif format == 'json':
                alerts.to_json(output_path, orient='records', indent=2)
            elif format == 'html':
                alerts.to_html(output_path, index=False)
            else:
                logger.warning(f"Unknown format: {format}")
                return
            
            logger.info(f"Exported {len(alerts)} alerts to {output_path}")
        
        except Exception as e:
            logger.error(f"Error exporting alerts: {str(e)}")
    
    def get_alert_summary(self, alerts: pd.DataFrame) -> Dict:
        """
        Get summary statistics of alerts.
        
        Args:
            alerts: DataFrame with alerts
            
        Returns:
            Dictionary with summary statistics
        """
        severity_counts = alerts['alert_severity'].value_counts().to_dict()
        
        summary = {
            'total_alerts': len(alerts),
            'severity_distribution': severity_counts,
            'critical_count': severity_counts.get('CRITICAL', 0),
            'high_count': severity_counts.get('HIGH', 0),
            'medium_count': severity_counts.get('MEDIUM', 0),
            'avg_risk_score': alerts['risk_score'].mean() if 'risk_score' in alerts.columns else 0,
            'avg_warning_prob': alerts['warning_probability'].mean() if 'warning_probability' in alerts.columns else 0,
        }
        
        logger.info(f"Alert summary: {summary}")
        
        return summary
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """
        Get recent alert history.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of alert records
        """
        return self.alert_history[-limit:]

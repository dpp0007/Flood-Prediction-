"""
Alert Manager for Flood Prediction System.

Handles alert generation and logging for HIGH risk predictions.

Data Source: Central Water Commission (CWC), Government of India
"""

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AlertManager:
    """
    Manages alert generation and logging.
    
    Features:
    - HIGH risk detection
    - Alert logging to file
    - Console alert printing
    - Alert statistics tracking
    """
    
    def __init__(self, alerts_dir: str = "data/outputs/alerts"):
        """
        Initialize alert manager.
        
        Args:
            alerts_dir: Directory for alert logs
        """
        self.alerts_dir = Path(alerts_dir)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AlertManager initialized. Alerts directory: {self.alerts_dir}")
    
    def process_predictions(self, predictions: pd.DataFrame,
                           station_metadata: pd.DataFrame) -> Dict[str, Any]:
        """
        Process predictions and generate alerts for HIGH risk.
        
        Args:
            predictions: Predictions DataFrame
            station_metadata: Station metadata DataFrame
            
        Returns:
            Dictionary with alert statistics
        """
        high_risk_predictions = predictions[
            predictions['risk_tier_name'].str.lower() == 'high'
        ]
        
        alerts = []
        
        for idx, row in high_risk_predictions.iterrows():
            try:
                station_id = row.get('station_id', 'UNKNOWN')
                station_name = row.get('station_name', 'Unknown Station')
                risk_score = float(row.get('risk_score', 0.0))
                
                # Get district from metadata
                district = self._get_district(station_id, station_metadata)
                
                # Create alert
                alert = self._create_alert(
                    station_id=station_id,
                    station_name=station_name,
                    district=district,
                    risk_score=risk_score
                )
                
                alerts.append(alert)
                
                # Print alert to console
                self._print_alert(alert)
            
            except Exception as e:
                logger.error(f"Error processing alert for {station_id}: {str(e)}")
                continue
        
        # Save alerts to file
        if alerts:
            self._save_alerts(alerts)
        
        return {
            'alerts_triggered': len(alerts),
            'alerts': alerts,
        }
    
    def _get_district(self, station_id: str, station_metadata: pd.DataFrame) -> str:
        """
        Get district for station.
        
        Args:
            station_id: Station identifier
            station_metadata: Station metadata DataFrame
            
        Returns:
            District name
        """
        if station_metadata is None or station_metadata.empty:
            return "Unknown"
        
        station_meta = station_metadata[
            station_metadata['station_id'] == station_id
        ]
        
        if station_meta.empty:
            return "Unknown"
        
        district = str(station_meta.iloc[0].get('district', 'Unknown')).strip()
        
        if not district or district.lower() in ['unknown', 'nan', '']:
            return "Unknown"
        
        return district
    
    def _create_alert(self, station_id: str, station_name: str,
                     district: str, risk_score: float) -> Dict[str, Any]:
        """
        Create alert dictionary.
        
        Args:
            station_id: Station identifier
            station_name: Station name
            district: District name
            risk_score: Risk score (0-1)
            
        Returns:
            Alert dictionary
        """
        timestamp = datetime.utcnow()
        
        alert = {
            'timestamp': timestamp.isoformat() + 'Z',
            'station_id': station_id,
            'station_name': station_name,
            'district': district,
            'risk_score': risk_score,
            'alert_type': 'HIGH_FLOOD_RISK',
            'message': self._format_alert_message(
                station_name=station_name,
                district=district,
                risk_score=risk_score,
                timestamp=timestamp
            )
        }
        
        return alert
    
    def _format_alert_message(self, station_name: str, district: str,
                             risk_score: float, timestamp: datetime) -> str:
        """
        Format alert message.
        
        Args:
            station_name: Station name
            district: District name
            risk_score: Risk score
            timestamp: Alert timestamp
            
        Returns:
            Formatted alert message
        """
        message = (
            f"ALERT: HIGH FLOOD RISK\n"
            f"Station: {station_name}\n"
            f"District: {district}\n"
            f"Risk Score: {risk_score:.1%}\n"
            f"Time: {timestamp.isoformat()}Z\n"
            f"Data Source: CWC (Govt. of India)\n"
            f"Disclaimer: Research/Demo use only"
        )
        
        return message
    
    def _print_alert(self, alert: Dict[str, Any]) -> None:
        """
        Print alert to console.
        
        Args:
            alert: Alert dictionary
        """
        print("\n" + "!"*70)
        print(alert['message'])
        print("!"*70 + "\n")
        
        logger.warning(f"[ALERT] HIGH FLOOD RISK: {alert['station_name']} ({alert['district']})")
    
    def _save_alerts(self, alerts: List[Dict[str, Any]]) -> None:
        """
        Save alerts to log file.
        
        Args:
            alerts: List of alert dictionaries
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        alert_file = self.alerts_dir / f"alerts_{timestamp}.log"
        
        try:
            with open(alert_file, 'w') as f:
                for alert in alerts:
                    f.write(alert['message'])
                    f.write("\n" + "-"*70 + "\n\n")
            
            logger.info(f"[ALERTS SAVED] {alert_file} ({len(alerts)} alerts)")
        
        except Exception as e:
            logger.error(f"Failed to save alerts: {str(e)}")

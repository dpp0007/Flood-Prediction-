"""
Autonomous Scheduler for Flood Prediction System.

Enables continuous operation with fixed-interval pipeline execution.

Data Source: Central Water Commission (CWC), Government of India
"""

import time
from datetime import datetime, timedelta
from typing import Callable, Optional
from pathlib import Path

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AutonomousScheduler:
    """
    Manages autonomous operation of the flood prediction pipeline.
    
    Features:
    - Fixed-interval execution
    - Continuous operation (until manually stopped)
    - Clear cycle logging
    - Graceful shutdown handling
    """
    
    def __init__(self, interval_minutes: int = 30):
        """
        Initialize scheduler.
        
        Args:
            interval_minutes: Interval between pipeline executions (default: 30)
        """
        self.interval_seconds = interval_minutes * 60
        self.interval_minutes = interval_minutes
        self.is_running = False
        self.cycle_count = 0
        
        logger.info("AutonomousScheduler initialized")
        logger.info(f"Execution interval: {interval_minutes} minutes ({self.interval_seconds} seconds)")
    
    def start(self, pipeline_func: Callable) -> None:
        """
        Start autonomous operation.
        
        Args:
            pipeline_func: Function to execute in each cycle
                          Should accept cycle_number as parameter
        """
        self.is_running = True
        self.cycle_count = 0
        
        logger.info("\n" + "="*70)
        logger.info("AUTONOMOUS SCHEDULER STARTED")
        logger.info("="*70)
        logger.info(f"Interval: {self.interval_minutes} minutes")
        logger.info(f"Press Ctrl+C to stop")
        logger.info("="*70 + "\n")
        
        try:
            while self.is_running:
                self.cycle_count += 1
                cycle_start = datetime.utcnow()
                
                # Print cycle header
                self._print_cycle_header(cycle_start)
                
                try:
                    # Execute pipeline
                    pipeline_func(cycle_number=self.cycle_count)
                    
                except Exception as e:
                    logger.error(f"Pipeline execution failed in cycle {self.cycle_count}: {str(e)}", exc_info=True)
                
                # Calculate next run time
                next_run = cycle_start + timedelta(seconds=self.interval_seconds)
                
                # Print cycle footer with next run time
                self._print_cycle_footer(next_run)
                
                # Wait for next cycle
                if self.is_running:
                    self._wait_for_next_cycle(next_run)
        
        except KeyboardInterrupt:
            logger.info("\n" + "="*70)
            logger.info("AUTONOMOUS SCHEDULER STOPPED (Ctrl+C)")
            logger.info("="*70)
            logger.info(f"Total cycles completed: {self.cycle_count}")
            logger.info("="*70 + "\n")
            self.is_running = False
    
    def _print_cycle_header(self, cycle_start: datetime) -> None:
        """Print cycle start header."""
        print("\n" + "-"*70)
        print(f"AUTONOMOUS PREDICTION CYCLE #{self.cycle_count}")
        print("-"*70)
        print(f"Cycle start time: {cycle_start.isoformat()}Z")
    
    def _print_cycle_footer(self, next_run: datetime) -> None:
        """Print cycle end footer with next run time."""
        print(f"Next run scheduled at: {next_run.isoformat()}Z")
        print("-"*70 + "\n")
    
    def _wait_for_next_cycle(self, next_run: datetime) -> None:
        """
        Wait until next cycle.
        
        Args:
            next_run: Datetime of next execution
        """
        while self.is_running:
            now = datetime.utcnow()
            
            if now >= next_run:
                break
            
            # Sleep in 1-second intervals to allow Ctrl+C
            time.sleep(1)
    
    def stop(self) -> None:
        """Stop autonomous operation."""
        logger.info("Stopping autonomous scheduler...")
        self.is_running = False


class CycleStatistics:
    """
    Tracks statistics for each prediction cycle.
    """
    
    def __init__(self, cycle_number: int):
        """
        Initialize cycle statistics.
        
        Args:
            cycle_number: Cycle number
        """
        self.cycle_number = cycle_number
        self.start_time = datetime.utcnow()
        
        self.total_stations = 0
        self.high_risk_count = 0
        self.medium_risk_count = 0
        self.low_risk_count = 0
        self.low_risk_json_generated = 0
        self.total_json_generated = 0
        self.alerts_triggered = 0
    
    def print_summary(self) -> None:
        """Print cycle summary."""
        print("----------------------------------------")
        print("AUTONOMOUS PREDICTION CYCLE")
        print("----------------------------------------")
        print(f"Cycle start time: {self.start_time.isoformat()}Z")
        print(f"Stations processed: {self.total_stations}")
        print(f"HIGH risk stations: {self.high_risk_count}")
        print(f"MEDIUM risk stations: {self.medium_risk_count}")
        print(f"LOW risk stations: {self.low_risk_count}")
        print(f"LOW risk JSON generated: {self.low_risk_json_generated} (max 3)")
        print(f"JSON files generated total: {self.total_json_generated}")
        print(f"Alerts triggered: {self.alerts_triggered}")
        print("----------------------------------------")

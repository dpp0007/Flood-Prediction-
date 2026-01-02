# main.py vs autonomous_main.py - Complete Comparison

**Date:** January 2, 2026

---

## Quick Summary

| Feature | main.py | autonomous_main.py |
|---------|---------|-------------------|
| **Purpose** | Single run | Continuous operation |
| **Execution** | One-time | Repeating (every 30 min) |
| **Scheduler** | None | Built-in scheduler |
| **Runs until** | Completes | Ctrl+C pressed |
| **Use case** | Testing/demo | Production 24/7 |
| **Pipeline** | Same | Same |

---

## Detailed Comparison

### 1. Purpose & Use Case

#### main.py
```
Purpose: Run the complete pipeline ONCE
Use case: 
  - Testing
  - Demonstration
  - One-time predictions
  - Manual execution
  - Debugging

Execution: Runs once and exits
```

#### autonomous_main.py
```
Purpose: Run the pipeline CONTINUOUSLY
Use case:
  - Production deployment
  - 24/7 monitoring
  - Scheduled predictions
  - Automated alerts
  - Long-running service

Execution: Runs indefinitely until Ctrl+C
```

---

### 2. Execution Flow

#### main.py
```
Start
  ↓
Load data
  ↓
Clean data
  ↓
Generate features
  ↓
Train models
  ↓
Generate predictions
  ↓
Generate alerts
  ↓
Save JSON
  ↓
Print summary
  ↓
Exit
```

#### autonomous_main.py
```
Start
  ↓
Initialize scheduler (30-minute interval)
  ↓
Loop (until Ctrl+C):
  ├─ Cycle 1:
  │  ├─ Load data
  │  ├─ Clean data
  │  ├─ Generate features
  │  ├─ Train models
  │  ├─ Generate predictions
  │  ├─ Generate alerts
  │  ├─ Save JSON
  │  └─ Print cycle summary
  │
  ├─ Wait 30 minutes
  │
  ├─ Cycle 2: (repeat)
  │
  └─ ... (continues until Ctrl+C)
  ↓
Graceful shutdown
  ↓
Exit
```

---

### 3. Scheduler

#### main.py
```python
# No scheduler
# Runs once and exits
def main():
    pipeline = CompleteFloodPredictionPipeline()
    results = pipeline.run(data_file)
    return 0 if results['status'] == 'success' else 1
```

#### autonomous_main.py
```python
# Built-in scheduler
# Runs continuously
def main():
    pipeline = AutonomousFloodPredictionPipeline()
    scheduler = AutonomousScheduler(interval_minutes=30)
    scheduler.start(pipeline_func=pipeline.run_cycle)
```

---

### 4. Cycle Management

#### main.py
```
Single execution:
- No cycle tracking
- No cycle statistics
- No next run time
- Runs once
```

#### autonomous_main.py
```
Multiple cycles:
- Cycle number tracking
- Cycle statistics (CycleStatistics class)
- Next run time calculation
- Runs indefinitely

Example:
  Cycle 1: 06:00 - 06:05
  Wait: 25 minutes
  Cycle 2: 06:30 - 06:35
  Wait: 25 minutes
  Cycle 3: 07:00 - 07:05
  ...
```

---

### 5. Alert Management

#### main.py
```python
# Standard alert generation
alerts = self.alert_gen.generate_alerts(predictions, station_metadata)
```

#### autonomous_main.py
```python
# Alert manager for HIGH risk
alert_manager = AlertManager()
alert_stats = alert_manager.process_predictions(predictions, station_metadata)

# Features:
# - HIGH risk alerts to console
# - Alert logs saved to data/outputs/alerts/
# - Alert statistics tracked
```

---

### 6. Output & Logging

#### main.py
```
Output:
  - JSON predictions
  - CSV predictions table
  - Alert CSV
  - Results metadata JSON
  - Validation report
  - Final execution summary

Logging:
  - Single execution log
  - No cycle tracking
```

#### autonomous_main.py
```
Output:
  - JSON predictions (per cycle)
  - CSV predictions table (per cycle)
  - Alert logs (per cycle)
  - Results metadata JSON (per cycle)
  - Cycle statistics
  - Alert logs in data/outputs/alerts/

Logging:
  - Per-cycle logs
  - Cycle start/end times
  - Next run time
  - Cycle statistics
  - Alert counts
```

---

### 7. Command Line Arguments

#### main.py
```bash
# Auto-detect latest data file
python main.py

# Use specific data file
python main.py --data-file data/raw/cwc_hydrograph_20260102_050433.csv
```

#### autonomous_main.py
```bash
# Default 30-minute interval
python autonomous_main.py

# Custom interval (60 minutes)
python autonomous_main.py --interval 60

# Custom interval (15 minutes)
python autonomous_main.py --interval 15
```

---

### 8. Pipeline Stages (IDENTICAL)

Both run the same pipeline:

```
1. Load raw CWC data
2. Clean & validate data
3. Generate 10 ML features
4. Create training labels
5. Train 3 ML models
6. Generate predictions
7. Generate alerts
8. Save JSON outputs
```

**Difference:** main.py runs once, autonomous_main.py repeats every 30 minutes

---

### 9. Shutdown Behavior

#### main.py
```
Normal exit:
  - Completes pipeline
  - Prints summary
  - Exits with code 0

Error exit:
  - Logs error
  - Exits with code 1
```

#### autonomous_main.py
```
Graceful shutdown (Ctrl+C):
  - Stops scheduler
  - Completes current cycle
  - Prints total cycles completed
  - Exits with code 0

Error handling:
  - Logs error
  - Continues to next cycle
  - Does NOT exit
```

---

### 10. Execution Time

#### main.py
```
Single run: ~77 seconds
Total time: ~77 seconds
```

#### autonomous_main.py
```
Per cycle: ~77 seconds
Interval: 30 minutes
Total time: Indefinite (until Ctrl+C)

Example (24 hours):
  - 48 cycles
  - ~3,696 seconds processing
  - ~83,904 seconds waiting
  - Total: 24 hours
```

---

## When to Use Each

### Use main.py When:
✅ Testing the pipeline  
✅ One-time predictions  
✅ Debugging issues  
✅ Manual execution  
✅ Demonstration  
✅ Development  

### Use autonomous_main.py When:
✅ Production deployment  
✅ 24/7 monitoring  
✅ Scheduled predictions  
✅ Automated alerts  
✅ Long-running service  
✅ Continuous operation  

---

## Code Structure Comparison

### main.py
```python
class CompleteFloodPredictionPipeline:
    def __init__(self)
    def load_raw_data(csv_path)
    def clean_data(raw_data)
    def generate_features(cleaned_data)
    def create_labels(features)
    def train_models(features, labels)
    def generate_predictions(features)
    def generate_alerts(predictions, metadata)
    def run(csv_path)                    # Single run
    def save_results(predictions, alerts, results, metadata)

def main():
    pipeline = CompleteFloodPredictionPipeline()
    results = pipeline.run(data_file)    # Run once
    return 0 if results['status'] == 'success' else 1
```

### autonomous_main.py
```python
class AutonomousFloodPredictionPipeline:
    def __init__(self)
    def load_raw_data(csv_path)
    def clean_data(raw_data)
    def generate_features(cleaned_data)
    def create_labels(features)
    def train_models(features, labels)
    def generate_predictions(features)
    def generate_alerts(predictions, metadata)
    def run_cycle(cycle_number)          # Per-cycle run
    def save_results(predictions, alerts, results, metadata)

class AutonomousScheduler:
    def __init__(interval_minutes)
    def start(pipeline_func)             # Continuous loop
    def _wait_for_next_cycle(next_run)

def main():
    pipeline = AutonomousFloodPredictionPipeline()
    scheduler = AutonomousScheduler(interval_minutes=30)
    scheduler.start(pipeline_func=pipeline.run_cycle)  # Run continuously
```

---

## Output Comparison

### main.py Output
```
======================================================================
FLOOD EARLY WARNING PREDICTION SYSTEM
======================================================================
Data Source: Central Water Commission (CWC), Government of India
Website: https://ffs.india-water.gov.in/
======================================================================

[Pipeline execution...]

FINAL EXECUTION SUMMARY
======================================================================
Status: SUCCESS
Raw records: 16800
Cleaned records: 16800
Features generated: 16800
Predictions: 16800
JSON files: 1200
Alerts: 50
======================================================================
```

### autonomous_main.py Output
```
======================================================================
AUTONOMOUS SCHEDULER STARTED
======================================================================
Interval: 30 minutes
Press Ctrl+C to stop
======================================================================

----------------------------------------------------------------------
AUTONOMOUS PREDICTION CYCLE #1
----------------------------------------------------------------------
Cycle start time: 2026-01-02T06:00:00.000000Z
Stations processed: 16800
HIGH risk stations: 50
MEDIUM risk stations: 100
LOW risk stations: 16650
JSON files generated total: 150
Alerts triggered: 50
Next run scheduled at: 2026-01-02T06:30:00.000000Z
----------------------------------------------------------------------

[Wait 30 minutes...]

----------------------------------------------------------------------
AUTONOMOUS PREDICTION CYCLE #2
----------------------------------------------------------------------
Cycle start time: 2026-01-02T06:30:00.000000Z
...
```

---

## File Locations (Same)

Both use the same directories:

```
data/
├── raw/                          # Input data
├── processed/                    # Cleaned data
├── features/                     # Features
├── models/                       # Trained models
└── outputs/
    ├── json_predictions/         # JSON output
    ├── alerts/                   # Alert logs
    └── tabular/                  # CSV predictions

logs/
└── flood_warning.log             # System logs
```

---

## Configuration

### main.py
```bash
# No configuration needed
python main.py
```

### autonomous_main.py
```bash
# Default (30 minutes)
python autonomous_main.py

# Custom interval
python autonomous_main.py --interval 60

# Configuration in config.yaml
scheduler:
  enabled: true
  interval_minutes: 30
  mode: "autonomous"
```

---

## Summary Table

| Aspect | main.py | autonomous_main.py |
|--------|---------|-------------------|
| **Execution** | Single run | Continuous loop |
| **Duration** | ~77 seconds | Indefinite |
| **Scheduler** | None | Built-in (30 min) |
| **Cycles** | 1 | Many |
| **Shutdown** | Auto | Ctrl+C |
| **Use case** | Testing | Production |
| **Pipeline** | Same | Same |
| **Output** | Single | Per-cycle |
| **Alerts** | Standard | Enhanced |
| **Logging** | Single | Per-cycle |

---

## Key Takeaway

**Pipeline logic is IDENTICAL**  
**Execution model is DIFFERENT**

- **main.py:** Run once, exit
- **autonomous_main.py:** Run continuously, repeat every 30 minutes

Choose based on your use case:
- **Testing/Demo:** Use main.py
- **Production/24/7:** Use autonomous_main.py

---

**Document Version:** 1.0  
**Date:** January 2, 2026  
**Status:** Complete

# Execution Modes Guide - Three Entry Points

**Date:** January 2, 2026

---

## Three Ways to Run the System

You have **3 different entry points**, each with a different purpose:

```
┌─────────────────────────────────────────────────────────────┐
│         Flood Early Warning Prediction System               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. main.py              → Single run (testing/demo)       │
│  2. main_2.py            → Single run + demo alerts        │
│  3. autonomous_main.py   → Continuous (production 24/7)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. main.py - Single Run (Production)

### Purpose
Run the complete pipeline **once** for real predictions

### Usage
```bash
python main.py
```

### Execution
```
Start → Load → Clean → Features → Train → Predict → Alerts → JSON → Exit
(~77 seconds total)
```

### Output
```
- JSON predictions (MEDIUM/HIGH risk)
- Alert logs
- Results metadata
- Validation report
- Final summary
```

### When to Use
✅ Testing the system  
✅ One-time predictions  
✅ Debugging  
✅ Manual execution  
✅ Demonstration  

### Example
```bash
# Run with auto-detected data
python main.py

# Run with specific data file
python main.py --data-file data/raw/cwc_hydrograph_20260102_050433.csv
```

---

## 2. main_2.py - Single Run + Demo Alerts

### Purpose
Run the pipeline **once** with **forced MEDIUM/HIGH alerts** for demonstration

### Usage
```bash
python main_2.py
```

### Execution
```
Start → Load → Clean → Features → Train → Predict → 
  ↓
  Check: Do MEDIUM/HIGH alerts exist?
  ├─ YES → Use natural predictions
  └─ NO → Force 1 HIGH + 2 MEDIUM alerts
  ↓
Alerts → JSON (with demo metadata) → Exit
(~77 seconds total)
```

### Output
```
- JSON predictions (with demo_mode flag)
- Alert logs
- Results metadata
- Demo alert summary
- Forced alert counts
```

### When to Use
✅ Testing alert system  
✅ Demonstrating capabilities  
✅ Integration testing  
✅ UI/UX testing  
✅ Stakeholder demos  

### Example
```bash
# Run with demo alerts
python main_2.py

# Check demo mode status
grep "DEMO MODE" logs/flood_warning.log

# View forced alerts in JSON
cat data/outputs/json_predictions/predictions_STN_*.json | jq '.metadata.demo_mode'
```

---

## 3. autonomous_main.py - Continuous (Production 24/7)

### Purpose
Run the pipeline **continuously** every 30 minutes for 24/7 monitoring

### Usage
```bash
python autonomous_main.py
```

### Execution
```
Start → Initialize Scheduler (30-minute interval)
  ↓
Loop (until Ctrl+C):
  ├─ Cycle 1: Load → Clean → Features → Train → Predict → Alerts → JSON
  ├─ Wait 30 minutes
  ├─ Cycle 2: Load → Clean → Features → Train → Predict → Alerts → JSON
  ├─ Wait 30 minutes
  ├─ Cycle 3: ...
  └─ ... (continues indefinitely)
  ↓
Graceful shutdown (Ctrl+C)
  ↓
Exit
```

### Output (Per Cycle)
```
- JSON predictions (MEDIUM/HIGH risk)
- Alert logs
- Results metadata
- Cycle statistics
- Next run time
```

### When to Use
✅ Production deployment  
✅ 24/7 monitoring  
✅ Scheduled predictions  
✅ Automated alerts  
✅ Long-running service  

### Example
```bash
# Run with default 30-minute interval
python autonomous_main.py

# Run with custom 60-minute interval
python autonomous_main.py --interval 60

# Run with custom 15-minute interval
python autonomous_main.py --interval 15

# Stop with Ctrl+C
# (Press Ctrl+C to gracefully shutdown)
```

### Cycle Output Example
```
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
```

---

## Comparison Table

| Feature | main.py | main_2.py | autonomous_main.py |
|---------|---------|-----------|-------------------|
| **Execution** | Single | Single | Continuous |
| **Duration** | ~77 sec | ~77 sec | Indefinite |
| **Runs** | 1 | 1 | Many (every 30 min) |
| **Alert forcing** | No | Yes (if needed) | No |
| **Demo mode** | No | Yes | No |
| **Scheduler** | No | No | Yes |
| **Use case** | Testing | Demo | Production |
| **Shutdown** | Auto | Auto | Ctrl+C |

---

## Pipeline Comparison

### All Three Use the Same Pipeline

```
Raw Data (CWC)
    ↓
Data Cleaning
    ↓
Feature Engineering (10 features)
    ↓
Label Creation
    ↓
Model Training (3 models)
    ↓
Predictions
    ↓
Alert Generation
    ↓
JSON Output
```

**Difference:** How many times it runs

- **main.py:** Once
- **main_2.py:** Once (+ demo alerts)
- **autonomous_main.py:** Repeatedly (every 30 min)

---

## Quick Decision Guide

### I want to test the system
→ Use **main.py**

### I want to demonstrate alerts
→ Use **main_2.py**

### I want production 24/7 monitoring
→ Use **autonomous_main.py**

### I want to debug an issue
→ Use **main.py**

### I want to test alert integration
→ Use **main_2.py**

### I want to deploy to production
→ Use **autonomous_main.py**

---

## File Locations (All Same)

```
Input:
  data/raw/cwc_hydrograph_*.csv

Output:
  data/outputs/json_predictions/predictions_*.json
  data/outputs/alerts/alerts_*.log
  data/processed/results_*.json
  logs/flood_warning.log
```

---

## Configuration

### main.py
```bash
python main.py
python main.py --data-file <path>
```

### main_2.py
```bash
python main_2.py
python main_2.py --data-file <path>
```

### autonomous_main.py
```bash
python autonomous_main.py                    # 30-minute interval
python autonomous_main.py --interval 60      # 60-minute interval
python autonomous_main.py --interval 15      # 15-minute interval
```

---

## Execution Examples

### Example 1: Test the system
```bash
# Run once to verify everything works
python main.py

# Check output
ls data/outputs/json_predictions/
cat logs/flood_warning.log | tail -50
```

### Example 2: Demo to stakeholders
```bash
# Run with guaranteed alerts
python main_2.py

# Show JSON with alerts
cat data/outputs/json_predictions/predictions_STN_*.json | jq '.prediction.risk_level'

# Show demo mode status
grep "DEMO MODE" logs/flood_warning.log
```

### Example 3: Deploy to production
```bash
# Run continuously (24/7)
python autonomous_main.py

# Monitor in another terminal
tail -f logs/flood_warning.log

# Stop when needed
# (Press Ctrl+C)
```

---

## Monitoring

### For main.py
```bash
# Check final summary
tail -20 logs/flood_warning.log
```

### For main_2.py
```bash
# Check demo mode
grep "DEMO MODE" logs/flood_warning.log

# Check forced alerts
grep "Forced" logs/flood_warning.log
```

### For autonomous_main.py
```bash
# Monitor in real-time
tail -f logs/flood_warning.log

# Check cycle statistics
grep "AUTONOMOUS PREDICTION CYCLE" logs/flood_warning.log

# Count cycles
grep "AUTONOMOUS PREDICTION CYCLE" logs/flood_warning.log | wc -l
```

---

## Performance

### main.py
- **Execution time:** ~77 seconds
- **Memory:** ~500 MB
- **Disk:** ~200 MB (JSON output)

### main_2.py
- **Execution time:** ~77 seconds
- **Memory:** ~500 MB
- **Disk:** ~200 MB (JSON output)

### autonomous_main.py (per cycle)
- **Execution time:** ~77 seconds per cycle
- **Memory:** ~500 MB
- **Disk:** ~200 MB per cycle
- **24-hour disk:** ~4.8 GB (48 cycles)

---

## Troubleshooting

### main.py issues
```bash
# Check logs
tail -100 logs/flood_warning.log

# Verify data file exists
ls data/raw/cwc_hydrograph_*.csv

# Check JSON output
ls data/outputs/json_predictions/
```

### main_2.py issues
```bash
# Check demo mode activated
grep "DEMO MODE" logs/flood_warning.log

# Verify forced alerts
grep "Forced" logs/flood_warning.log

# Check JSON metadata
cat data/outputs/json_predictions/predictions_STN_*.json | jq '.metadata.demo_mode'
```

### autonomous_main.py issues
```bash
# Check scheduler started
grep "AUTONOMOUS SCHEDULER STARTED" logs/flood_warning.log

# Check cycle execution
grep "AUTONOMOUS PREDICTION CYCLE" logs/flood_warning.log

# Check for errors
grep "ERROR" logs/flood_warning.log

# Monitor in real-time
tail -f logs/flood_warning.log
```

---

## Summary

| Mode | Command | Use Case | Duration |
|------|---------|----------|----------|
| **Testing** | `python main.py` | Single run | ~77 sec |
| **Demo** | `python main_2.py` | With alerts | ~77 sec |
| **Production** | `python autonomous_main.py` | 24/7 | Indefinite |

**All three use the same pipeline and produce the same quality predictions.**

---

**Document Version:** 1.0  
**Date:** January 2, 2026  
**Status:** Complete

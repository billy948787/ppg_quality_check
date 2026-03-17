# PPG Quality Check

Signal quality assessment for PPG (photoplethysmography) data. Supports multi-channel CSV files and produces per-channel analysis reports plus a cross-channel comparison.

## What it does

For each signal channel found in the CSV:
- Bandpass filter → smoothing → Tukey taper
- Peak / valley detection and RR interval extraction
- Standard SQI metrics (kurtosis, skewness, entropy, SNR, perfusion, ZCR)
- HRV time & frequency domain, Poincaré, DTW template matching
- Composite SQI score (0–100, grades A–F) + rule-based quality label (GOOD / FAIR / POOR)
- 30-second segment analysis

If multiple channels are present, a multi-band comparison dashboard is generated.

## Setup

```bash
uv sync
```

## Usage

```bash
# specify a file
python main.py s17_run.csv
python main.py Raw_data.csv

# common options
python main.py s17_run.csv --no-show        # save plots but don't display
python main.py s17_run.csv --no-save        # display only, don't write files
python main.py s17_run.csv --out-dir plots/ # save to a specific directory
python main.py s17_run.csv --fs 200         # override sampling rate
python main.py s17_run.csv --seg-s 60       # 60-second segments
```

## CSV format

Column names are auto-detected — no configuration needed.

| What | Detected patterns |
|------|-------------------|
| Signal | `pleth`, `ppg`, `ir`, `red`, `green`, `nir` |
| Timestamp | `time`, `timestamp`, `sensor_timestamp`, `ts`, `datetime` |

Multiple matching signal columns → each is analysed as a separate channel.  
Timestamps can be in nanoseconds, microseconds, milliseconds, seconds, or datetime strings.

## Output

| File | Description |
|------|-------------|
| `ppg_preprocessing_<ch>.png` | Raw → filtered → smoothed for each channel |
| `ppg_analysis_report_<ch>.png` | 11-panel full report (peaks, HRV, segments, score gauge) |
| `ppg_multiband_comparison.png` | Side-by-side comparison across all channels |

## Package structure

```
ppg/
├── config.py          PPGConfig dataclass – all tunable parameters
├── loader.py          CSV loading, column auto-detection, resampling
├── preprocessing.py   Bandpass → smooth → taper
├── peaks.py           Valley-based fiducial detection
├── features/          Standard SQI, HRV, DTW, waveform, r-peak SQI
├── scoring.py         Composite SQI score
├── rules.py           Rule-based quality labelling
├── segments.py        Per-segment analysis
├── pipeline.py        Single-channel orchestration
└── visualization.py   All matplotlib plots
```

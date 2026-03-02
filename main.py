"""
Comprehensive PPG Signal Quality Analysis using vital_sqi
==========================================================
This script performs a full signal quality analysis pipeline on raw PPG data:
1. Signal preprocessing (bandpass filter, smoothing, tapering)
2. Standard SQI metrics (kurtosis, skewness, entropy, SNR, perfusion, crossing rates)
3. Peak detection & NN interval extraction
4. HRV time domain metrics (SDNN, SDSD, RMSSD, CVSD, CVNN, pNN50, etc.)
5. Heart rate statistics (mean, median, min, max, std, range)
6. HRV frequency domain metrics (peak freq, absolute/log/relative/normalized power, LF/HF)
7. Poincaré features (SD1, SD2, area, ratio)
8. DTW template matching SQI
9. R-peak based SQI (ectopic ratio, correlogram, MSQ)
10. Waveform SQI (band energy, LF energy)
11. Full HRV analysis via hrvanalysis
12. Segment-level SQI analysis (30s segments)
13. Comprehensive visualization
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, find_peaks

# ─── vital_sqi imports ──────────────────────────────────────────────
from vital_sqi.preprocess.preprocess_signal import smooth_signal, taper_signal
from vital_sqi.common.band_filter import BandpassFilter
from vital_sqi.common.rpeak_detection import PeakDetector
from vital_sqi.common.utils import get_nn

# Standard SQI
from vital_sqi.sqi.standard_sqi import (
    kurtosis_sqi, skewness_sqi, entropy_sqi,
    signal_to_noise_sqi, perfusion_sqi,
    zero_crossings_rate_sqi, mean_crossing_rate_sqi,
)

# HRV SQI
from vital_sqi.sqi.hrv_sqi import (
    nn_mean_sqi, sdnn_sqi, sdsd_sqi, rmssd_sqi,
    cvsd_sqi, cvnn_sqi, mean_nn_sqi, median_nn_sqi,
    pnn_sqi, hr_mean_sqi, hr_median_sqi, hr_min_sqi,
    hr_max_sqi, hr_std_sqi, hr_range_sqi,
    peak_frequency_sqi, absolute_power_sqi, log_power_sqi,
    relative_power_sqi, normalized_power_sqi, lf_hf_ratio_sqi,
    poincare_features_sqi, get_all_features_hrva,
)

# DTW SQI
from vital_sqi.sqi.dtw_sqi import dtw_sqi

# R-peak based SQI
from vital_sqi.sqi.rpeaks_sqi import ectopic_sqi, correlogram_sqi, msq_sqi

# Waveform SQI
from vital_sqi.sqi.waveform_sqi import band_energy_sqi, lf_energy_sqi

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════
# 0. Load & inspect raw data (bypass PPG_reader for correct sampling rate)
# ════════════════════════════════════════════════════════════════════
file_in = os.path.abspath("Raw_data.csv")

# Read CSV directly — PPG_reader's calculate_sampling_rate uses min(diff)
# which picks up timestamp jitter (1ms) → bogus 1001 Hz
df_raw = pd.read_csv(file_in)
df_raw.columns = df_raw.columns.str.strip()

raw_signal = df_raw["data"].values.astype(float)
timestamps_ms = df_raw["timestamp"].values.astype(float)

# Compute duration and detect timestamp issues
ts_diffs_ms = np.diff(timestamps_ms)
duration_sec = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
n_duplicates = np.sum(ts_diffs_ms == 0)

# Resample to uniform time grid at 100 Hz
# The raw timestamps are irregular (40%+ duplicates, variable intervals).
# Interpolating onto a uniform grid gives the same effect as fuck.py's
# assumption of equally-spaced samples at 100 Hz.
sampling_rate = 100
t_raw_sec = (timestamps_ms - timestamps_ms[0]) / 1000.0  # relative time in seconds
t_uniform = np.arange(0, duration_sec, 1.0 / sampling_rate)  # uniform grid
raw_signal_resampled = np.interp(t_uniform, t_raw_sec, raw_signal)

N_original = len(raw_signal)
raw_signal = raw_signal_resampled  # use resampled signal from here on
N = len(raw_signal)

print("=" * 70)
print("PPG Signal Quality Analysis Report")
print("=" * 70)
print(f"  Original samples : {N_original}")
print(f"  Duplicate ts     : {n_duplicates} ({100*n_duplicates/len(ts_diffs_ms):.1f}%)")
print(f"  Resampled to     : {N} samples @ {sampling_rate} Hz (uniform grid)")
print(f"  Duration         : {duration_sec:.1f} s")
print(f"  Raw range        : [{raw_signal.min():.0f}, {raw_signal.max():.0f}]")
print(f"  Raw mean (DC)    : {raw_signal.mean():.0f}")

# ════════════════════════════════════════════════════════════════════
# 1. Signal Preprocessing
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("1. Signal Preprocessing")
print("─" * 70)

# Step 1a: Bandpass filter (0.5 – 5 Hz) — isolate PPG heart rate band
# No need to manually remove DC offset — bandpass inherently rejects DC (0 Hz)
# Using 5 Hz highcut (not 8 Hz) to match PPG physiological range (0.5-4 Hz)
lowcut, highcut = 0.5, 5.0
nyq = 0.5 * sampling_rate
b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
filtered_signal = filtfilt(b, a, raw_signal)
print(f"  [1a] Bandpass      : {lowcut}–{highcut} Hz, 4th order Butterworth")
print(f"       Filtered range: [{filtered_signal.min():.2f}, {filtered_signal.max():.2f}]")

# Step 1c: Smoothing (Hanning window)
smoothed_signal = smooth_signal(filtered_signal, window_len=5, window="hanning")
smoothed_signal = smoothed_signal[:N]
print(f"  [1b] Smoothing     : Hanning window, len=5")

# Step 1d: Tapering (for template matching)
tapered_signal = taper_signal(smoothed_signal)
print(f"  [1c] Tapering      : Tukey window (for DTW template matching)")

# The main processed signal for analysis
processed_signal = filtered_signal

# Preprocessing quality checks
snr_check = np.std(processed_signal) / np.std(processed_signal - smoothed_signal) if np.std(processed_signal - smoothed_signal) > 0 else float('inf')
print(f"\n  Preprocessing Quality:")
print(f"    Processed range  : [{processed_signal.min():.2f}, {processed_signal.max():.2f}]")
print(f"    Processed std    : {np.std(processed_signal):.2f}")
print(f"    Smoothing SNR    : {snr_check:.2f}")

# ────────────────────────────────────────────────────────────────────
# Preprocessing Verification Plot
# ────────────────────────────────────────────────────────────────────
print("\n  Generating preprocessing verification plot...")

time_axis = np.arange(N) / sampling_rate

fig_pre, axes_pre = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
fig_pre.suptitle("PPG Preprocessing Verification", fontsize=16, fontweight="bold")

# Panel 1: Raw signal
axes_pre[0].plot(time_axis, raw_signal, linewidth=0.6, color="steelblue")
axes_pre[0].set_title(f"(a) Raw PPG Signal (range: {raw_signal.min():.0f}–{raw_signal.max():.0f})", fontsize=11)
axes_pre[0].set_ylabel("ADC Value")
axes_pre[0].grid(True, alpha=0.3)

# Panel 2: After bandpass filter
axes_pre[1].plot(time_axis, filtered_signal, linewidth=0.6, color="crimson")
axes_pre[1].set_title(f"(b) After Bandpass Filter ({lowcut}–{highcut} Hz)", fontsize=11)
axes_pre[1].set_ylabel("Amplitude")
axes_pre[1].grid(True, alpha=0.3)

# Panel 3: Filtered vs Smoothed (final processed)
axes_pre[2].plot(time_axis, filtered_signal, linewidth=0.6, alpha=0.4, color="crimson", label="Filtered")
axes_pre[2].plot(time_axis, smoothed_signal, linewidth=1.0, color="green", label="Smoothed")
axes_pre[2].set_title("(c) Filtered vs Smoothed (final)", fontsize=11)
axes_pre[2].set_ylabel("Amplitude")
axes_pre[2].set_xlabel("Time (s)")
axes_pre[2].legend(loc="upper right")
axes_pre[2].grid(True, alpha=0.3)

plt.tight_layout()
preprocess_fig_path = os.path.join(os.path.dirname(file_in), "..", "ppg_quality_check", "ppg_preprocessing_verification.png")
plt.savefig(preprocess_fig_path, dpi=150, bbox_inches="tight")
print(f"  ✓ Saved: ppg_preprocessing_verification.png")
plt.show()

# ════════════════════════════════════════════════════════════════════
# 2. Standard SQI Metrics
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("2. Standard SQI Metrics (Statistical Domain)")
print("─" * 70)

standard_results = {}

standard_results["Kurtosis"] = kurtosis_sqi(processed_signal)
standard_results["Skewness"] = skewness_sqi(processed_signal)
standard_results["Entropy"] = entropy_sqi(processed_signal)
standard_results["SNR"] = signal_to_noise_sqi(processed_signal)
standard_results["Perfusion Index (%)"] = perfusion_sqi(raw_signal, processed_signal)
standard_results["Zero Crossing Rate"] = zero_crossings_rate_sqi(processed_signal)
standard_results["Mean Crossing Rate"] = mean_crossing_rate_sqi(processed_signal)

for name, val in standard_results.items():
    print(f"  {name:30s}: {val:.6f}")

# ════════════════════════════════════════════════════════════════════
# 3. Peak Detection
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("3. Peak Detection")
print("─" * 70)

detector = PeakDetector(wave_type="ppg", fs=sampling_rate)

# Try multiple detectors and pick the best result
detector_types = {
    1: "Adaptive Threshold",
    2: "Count Orig",
    3: "Clusterer",
    4: "Slope Sum",
    5: "Moving Average",
    6: "Scipy Default",
    7: "Billauer",
}

def _score_detector(p, t, sr, label):
    """Score a peak detector result by physiological plausibility."""
    n_peaks, n_troughs = len(p), len(t)
    score, hr_est = 0, 0
    if n_peaks > 2:
        rr_ms = np.diff(p) * (1000 / sr)
        hr_est = 60000 / np.mean(rr_ms)
        rr_cv = np.std(rr_ms) / np.mean(rr_ms) if np.mean(rr_ms) > 0 else 999
        # Continuous HR scoring: ideal range 50-150 bpm, penalize deviation
        if 50 <= hr_est <= 150:
            score += 100
        elif 40 <= hr_est <= 180:
            score += 70
        elif 30 <= hr_est <= 200:
            score += 30
        # Bonus for troughs (needed for DTW, beat extraction)
        if n_troughs > 0:
            score += 20
        # Lower RR variability = more consistent detection
        score += max(0, 30 - rr_cv * 30)
        score += min(n_peaks / 10, 5)
    status = f"HR≈{hr_est:.0f}bpm, score={score:.0f}" if n_peaks > 2 else "too few"
    print(f"    {label:30s}: {n_peaks} peaks, {n_troughs} troughs ({status})")
    return score, hr_est

best_peaks, best_troughs, best_name, best_score = [], [], "None", -1
print("  Trying multiple peak detectors:")

# vital_sqi detectors
for dt_id, dt_name in detector_types.items():
    try:
        p, t = detector.ppg_detector(processed_signal, detector_type=dt_id)
        score, _ = _score_detector(p, t, sampling_rate, f"[{dt_id}] {dt_name}")
        if score > best_score:
            best_peaks, best_troughs, best_name, best_score = list(p), list(t), dt_name, score
    except Exception as e:
        print(f"    [{dt_id}] {dt_name:25s}: Error - {e}")

# Additional: scipy find_peaks with physiological min distance
# Min distance = 0.3s (max 200 bpm), find both peaks and troughs
min_dist = int(sampling_rate * 0.3)  # 200 bpm max
sp_peaks, _ = find_peaks(processed_signal, distance=min_dist)
sp_troughs, _ = find_peaks(-processed_signal, distance=min_dist)
score, _ = _score_detector(sp_peaks, sp_troughs, sampling_rate, "[+] Scipy find_peaks (d=0.3s)")
if score > best_score:
    best_peaks, best_troughs, best_name, best_score = list(sp_peaks), list(sp_troughs), "Scipy find_peaks", score

peaks = np.array(best_peaks)
troughs = np.array(best_troughs)
print(f"\n  ✓ Best detector : {best_name}")
print(f"  Peaks detected  : {len(peaks)}")
print(f"  Troughs detected: {len(troughs)}")

rr_intervals = np.array([])
if len(peaks) > 1:
    rr_intervals = np.diff(peaks) * (1000 / sampling_rate)  # ms
    print(f"  RR intervals   : {len(rr_intervals)}")
    print(f"  RR mean        : {np.mean(rr_intervals):.1f} ms")
    print(f"  RR std         : {np.std(rr_intervals):.1f} ms")
    print(f"  RR range       : [{np.min(rr_intervals):.1f}, {np.max(rr_intervals):.1f}] ms")
else:
    print("  ⚠ No peak detector found more than 1 peak")

# ════════════════════════════════════════════════════════════════════
# 4. HRV Time Domain Metrics
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("4. HRV Time Domain Metrics")
print("─" * 70)

hrv_time_results = {}

if len(peaks) > 2:
    nn_intervals = rr_intervals  # using RR as NN for quality assessment

    hrv_time_results["Mean NN (ms)"] = mean_nn_sqi(nn_intervals)
    hrv_time_results["Median NN (ms)"] = median_nn_sqi(nn_intervals)
    hrv_time_results["SDNN (ms)"] = sdnn_sqi(nn_intervals)
    hrv_time_results["SDSD (ms)"] = sdsd_sqi(nn_intervals)
    hrv_time_results["RMSSD (ms)"] = rmssd_sqi(nn_intervals)
    hrv_time_results["CVSD"] = cvsd_sqi(nn_intervals)
    hrv_time_results["CVNN"] = cvnn_sqi(nn_intervals)
    hrv_time_results["pNN50 (%)"] = pnn_sqi(nn_intervals, exceed=50)
    hrv_time_results["pNN20 (%)"] = pnn_sqi(nn_intervals, exceed=20)

    for name, val in hrv_time_results.items():
        print(f"  {name:30s}: {val:.4f}")
else:
    print("  ⚠ Not enough peaks for HRV analysis")

# ════════════════════════════════════════════════════════════════════
# 5. Heart Rate Statistics
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("5. Heart Rate Statistics")
print("─" * 70)

hr_results = {}

if len(peaks) > 2:
    hr_results["HR Mean (bpm)"] = hr_mean_sqi(nn_intervals)
    hr_results["HR Median (bpm)"] = hr_median_sqi(nn_intervals)
    hr_results["HR Min (bpm)"] = hr_min_sqi(nn_intervals)
    hr_results["HR Max (bpm)"] = hr_max_sqi(nn_intervals)
    hr_results["HR Std (bpm)"] = hr_std_sqi(nn_intervals)
    hr_results["HR Out-of-Range (%, 40-200)"] = hr_range_sqi(nn_intervals)

    for name, val in hr_results.items():
        print(f"  {name:30s}: {val:.2f}")
else:
    print("  ⚠ Not enough peaks for HR analysis")

# ════════════════════════════════════════════════════════════════════
# 6. HRV Frequency Domain Metrics
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("6. HRV Frequency Domain Metrics")
print("─" * 70)

hrv_freq_results = {}

if len(peaks) > 10:
    try:
        hrv_freq_results["Peak Frequency (LF)"] = peak_frequency_sqi(nn_intervals, f_min=0.04, f_max=0.15)
        hrv_freq_results["Peak Frequency (HF)"] = peak_frequency_sqi(nn_intervals, f_min=0.15, f_max=0.40)
        hrv_freq_results["Absolute Power (LF)"] = absolute_power_sqi(nn_intervals, f_min=0.04, f_max=0.15)
        hrv_freq_results["Absolute Power (HF)"] = absolute_power_sqi(nn_intervals, f_min=0.15, f_max=0.40)
        hrv_freq_results["Log Power (LF)"] = log_power_sqi(nn_intervals, f_min=0.04, f_max=0.15)
        hrv_freq_results["Log Power (HF)"] = log_power_sqi(nn_intervals, f_min=0.15, f_max=0.40)
        hrv_freq_results["Relative Power (LF)"] = relative_power_sqi(nn_intervals, f_min=0.04, f_max=0.15)
        hrv_freq_results["Relative Power (HF)"] = relative_power_sqi(nn_intervals, f_min=0.15, f_max=0.40)
        hrv_freq_results["Normalized Power"] = normalized_power_sqi(nn_intervals)
        hrv_freq_results["LF/HF Ratio"] = lf_hf_ratio_sqi(nn_intervals)

        for name, val in hrv_freq_results.items():
            print(f"  {name:30s}: {val:.6f}")
    except Exception as e:
        print(f"  ⚠ Frequency domain error: {e}")
else:
    print("  ⚠ Not enough peaks for frequency domain analysis")

# ════════════════════════════════════════════════════════════════════
# 7. Poincaré Features
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("7. Poincaré Features")
print("─" * 70)

if len(peaks) > 3:
    poincare = poincare_features_sqi(nn_intervals)
    for name, val in poincare.items():
        print(f"  {name:40s}: {val:.6f}")
else:
    print("  ⚠ Not enough data for Poincaré analysis")

# ════════════════════════════════════════════════════════════════════
# 8. DTW Template Matching SQI
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("8. DTW Template Matching SQI")
print("─" * 70)

dtw_results = {}
template_names = {
    0: "Nonlinear Dynamic System",
    1: "Dual Double Frequency",
    2: "Absolute Dual Skewness",
}

# Use individual beats for DTW
if len(peaks) >= 3 and len(troughs) >= 2:
    # Extract a representative beat (mid-signal)
    mid_idx = len(peaks) // 2
    if mid_idx > 0 and mid_idx < len(peaks) - 1:
        beat_start = troughs[min(mid_idx - 1, len(troughs) - 1)]
        beat_end = troughs[min(mid_idx, len(troughs) - 1)]
        beat = processed_signal[beat_start:beat_end]

        if len(beat) > 10:
            for tmpl_type, tmpl_name in template_names.items():
                try:
                    cost = dtw_sqi(beat, template_type=tmpl_type, template_size=100)
                    dtw_results[tmpl_name] = cost
                    print(f"  DTW [{tmpl_name:30s}]: {cost:.6f}")
                except Exception as e:
                    print(f"  DTW [{tmpl_name:30s}]: Error - {e}")
        else:
            print("  ⚠ Beat too short for DTW analysis")
    else:
        print("  ⚠ Could not extract representative beat")
else:
    print("  ⚠ Not enough peaks/troughs for DTW analysis")

# Compute average DTW across multiple beats
if len(peaks) >= 5 and len(troughs) >= 4:
    print("\n  --- Per-Beat DTW (Template 0: Nonlinear Dynamic) ---")
    beat_dtw_scores = []
    n_beats_to_check = min(10, len(troughs) - 1)
    for i in range(n_beats_to_check):
        try:
            b_start = troughs[i]
            b_end = troughs[i + 1]
            beat_seg = processed_signal[b_start:b_end]
            if len(beat_seg) > 10:
                score = dtw_sqi(beat_seg, template_type=0, template_size=100)
                beat_dtw_scores.append(score)
        except:
            pass
    if beat_dtw_scores:
        print(f"  Beats analyzed : {len(beat_dtw_scores)}")
        print(f"  DTW Mean       : {np.mean(beat_dtw_scores):.6f}")
        print(f"  DTW Std        : {np.std(beat_dtw_scores):.6f}")
        print(f"  DTW Min        : {np.min(beat_dtw_scores):.6f}")
        print(f"  DTW Max        : {np.max(beat_dtw_scores):.6f}")

# ════════════════════════════════════════════════════════════════════
# 9. R-Peak Based SQI
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("9. R-Peak Based SQI")
print("─" * 70)

rpeak_results = {}

try:
    ectopic_ratio = ectopic_sqi(
        processed_signal, rule_index=1,
        sample_rate=sampling_rate, rpeak_detector=0,
        wave_type="ppg"
    )
    rpeak_results["Ectopic Ratio (Malik)"] = ectopic_ratio
    print(f"  {'Ectopic Ratio (Malik)':30s}: {ectopic_ratio:.6f}")
except Exception as e:
    print(f"  Ectopic SQI Error: {e}")

try:
    corr_features = correlogram_sqi(
        processed_signal, sample_rate=sampling_rate,
        time_lag=3, n_selection=3
    )
    rpeak_results["Correlogram Features"] = corr_features
    print(f"  {'Correlogram Peaks (idx)':30s}: {corr_features[:3] if len(corr_features) >= 3 else corr_features}")
    print(f"  {'Correlogram Peaks (val)':30s}: {corr_features[3:] if len(corr_features) > 3 else 'N/A'}")
except Exception as e:
    print(f"  Correlogram SQI Error: {e}")

try:
    msq_value = msq_sqi(processed_signal, peak_detector_1=7, peak_detector_2=6, wave_type="ppg")
    rpeak_results["MSQ (Multi-Detector)"] = msq_value
    print(f"  {'MSQ (Detector Consistency)':30s}: {msq_value:.6f}")
except Exception as e:
    print(f"  MSQ SQI Error: {e}")

# ════════════════════════════════════════════════════════════════════
# 10. Waveform SQI (Energy-based)
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("10. Waveform SQI (Energy-based)")
print("─" * 70)

waveform_results = {}

try:
    total_energy = band_energy_sqi(processed_signal, sampling_rate=sampling_rate, band=None)
    waveform_results["Total Band Energy"] = total_energy
    print(f"  {'Total Band Energy':30s}: {total_energy:.6f}")
except Exception as e:
    print(f"  Band Energy Error: {e}")

try:
    lf_energy = lf_energy_sqi(processed_signal, sampling_rate=sampling_rate, band=[0, 0.5])
    waveform_results["LF Energy (0-0.5Hz)"] = lf_energy
    print(f"  {'LF Energy (0-0.5 Hz)':30s}: {lf_energy:.6f}")
except Exception as e:
    print(f"  LF Energy Error: {e}")

try:
    ppg_band_energy = band_energy_sqi(processed_signal, sampling_rate=sampling_rate, band=[0.5, 4.0])
    waveform_results["PPG Band Energy (0.5-4Hz)"] = ppg_band_energy
    print(f"  {'PPG Band Energy (0.5-4 Hz)':30s}: {ppg_band_energy:.6f}")
except Exception as e:
    print(f"  PPG Band Energy Error: {e}")

try:
    hf_energy = band_energy_sqi(processed_signal, sampling_rate=sampling_rate, band=[4.0, 8.0])
    waveform_results["HF Energy (4-8Hz)"] = hf_energy
    print(f"  {'HF Energy (4-8 Hz)':30s}: {hf_energy:.6f}")
except Exception as e:
    print(f"  HF Energy Error: {e}")

# ════════════════════════════════════════════════════════════════════
# 11. Full HRV Analysis (via hrvanalysis)
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("11. Full HRV Analysis (hrvanalysis)")
print("─" * 70)

try:
    time_feats, freq_feats, geom_feats, csi_cvi_feats = get_all_features_hrva(
        processed_signal, sample_rate=sampling_rate, rpeak_method=0, wave_type="ppg"
    )

    if time_feats:
        print("\n  [Time Domain Features]")
        for k, v in time_feats.items():
            print(f"    {k:35s}: {v}")

    if freq_feats:
        print("\n  [Frequency Domain Features]")
        for k, v in freq_feats.items():
            print(f"    {k:35s}: {v}")

    if geom_feats:
        print("\n  [Geometrical Features]")
        for k, v in geom_feats.items():
            print(f"    {k:35s}: {v}")

    if csi_cvi_feats:
        print("\n  [CSI/CVI Features]")
        for k, v in csi_cvi_feats.items():
            print(f"    {k:35s}: {v}")

except Exception as e:
    print(f"  ⚠ Full HRV Analysis Error: {e}")

# ════════════════════════════════════════════════════════════════════
# 12. Segment-Level Analysis (30s segments)
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("12. Segment-Level SQI Analysis (30s segments)")
print("─" * 70)

segment_duration = 30  # seconds
segment_size = segment_duration * sampling_rate
n_segments = int(np.ceil(N / segment_size))  # include the last partial segment

segment_sqis = []

if n_segments < 1:
    # Signal too short, use whole signal as single segment
    segment_duration = int(N / sampling_rate)
    segment_size = N
    n_segments = 1
    print(f"  Signal shorter than 30s, using full {segment_duration}s as single segment")

print(f"  Segment duration : {segment_duration}s")
print(f"  Number of segments: {n_segments}\n")

for seg_idx in range(n_segments):
    start = seg_idx * segment_size
    end = min(start + segment_size, N)
    seg = processed_signal[start:end]

    seg_result = {
        "segment": seg_idx + 1,
        "start_s": start / sampling_rate,
        "end_s": end / sampling_rate,
    }

    # Standard SQI per segment
    seg_result["kurtosis"] = kurtosis_sqi(seg)
    seg_result["skewness"] = skewness_sqi(seg)
    seg_result["entropy"] = entropy_sqi(seg)
    seg_result["snr"] = signal_to_noise_sqi(seg)
    seg_result["zcr"] = zero_crossings_rate_sqi(seg)
    seg_result["mcr"] = mean_crossing_rate_sqi(seg)
    seg_result["perfusion"] = perfusion_sqi(raw_signal[start:end], seg)

    # Peak-based per segment
    try:
        seg_peaks, seg_troughs = detector.ppg_detector(seg, detector_type=1)
        seg_result["n_peaks"] = len(seg_peaks)
        if len(seg_peaks) > 2:
            seg_rr = np.diff(seg_peaks) * (1000 / sampling_rate)
            seg_result["hr_mean"] = hr_mean_sqi(seg_rr)
            seg_result["rmssd"] = rmssd_sqi(seg_rr)
            seg_result["sdnn"] = sdnn_sqi(seg_rr)
        else:
            seg_result["hr_mean"] = np.nan
            seg_result["rmssd"] = np.nan
            seg_result["sdnn"] = np.nan
    except:
        seg_result["n_peaks"] = 0
        seg_result["hr_mean"] = np.nan
        seg_result["rmssd"] = np.nan
        seg_result["sdnn"] = np.nan

    segment_sqis.append(seg_result)

    print(f"  Segment {seg_idx+1} [{seg_result['start_s']:.0f}s – {seg_result['end_s']:.0f}s]:")
    print(f"    Kurtosis={seg_result['kurtosis']:.3f}  Skewness={seg_result['skewness']:.3f}  "
          f"Entropy={seg_result['entropy']:.3f}  SNR={seg_result['snr']:.3f}")
    print(f"    ZCR={seg_result['zcr']:.3f}  MCR={seg_result['mcr']:.3f}  "
          f"Perfusion={seg_result['perfusion']:.3f}%")
    print(f"    Peaks={seg_result['n_peaks']}  HR={seg_result['hr_mean']:.0f}bpm  "
          f"RMSSD={seg_result['rmssd']:.1f}ms  SDNN={seg_result['sdnn']:.1f}ms"
          if not np.isnan(seg_result.get('hr_mean', np.nan)) else
          f"    Peaks={seg_result['n_peaks']}  (insufficient for HR/HRV)")

# ════════════════════════════════════════════════════════════════════
# 13. Visualization
# ════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("13. Generating Visualizations...")
print("─" * 70)

time_axis = np.arange(N) / sampling_rate

fig = plt.figure(figsize=(20, 24))
gs = gridspec.GridSpec(6, 2, hspace=0.4, wspace=0.3)

# ─── Plot 1: Raw vs Filtered  ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time_axis, raw_signal, alpha=0.5, linewidth=0.5, label="Raw PPG", color="steelblue")
ax1.plot(time_axis, processed_signal, linewidth=1.0, label="Filtered PPG", color="crimson")
ax1.set_title("Raw vs Filtered PPG Signal", fontsize=14, fontweight="bold")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Amplitude")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# ─── Plot 2: Filtered + Detected Peaks ─────────────────────────
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(time_axis, processed_signal, linewidth=0.8, color="steelblue", label="Filtered PPG")
if len(peaks) > 0:
    ax2.scatter(np.array(peaks) / sampling_rate, processed_signal[peaks],
                color="red", s=20, zorder=5, label=f"Peaks ({len(peaks)})")
if len(troughs) > 0:
    ax2.scatter(np.array(troughs) / sampling_rate, processed_signal[troughs],
                color="green", s=15, zorder=5, label=f"Troughs ({len(troughs)})")
ax2.set_title("Peak Detection Results", fontsize=14, fontweight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude")
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

# ─── Plot 3: Standard SQI Bar Chart ────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
sqi_names = list(standard_results.keys())
sqi_values = list(standard_results.values())
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sqi_names)))
bars = ax3.barh(sqi_names, sqi_values, color=colors, edgecolor="white")
ax3.set_title("Standard SQI Metrics", fontsize=12, fontweight="bold")
ax3.set_xlabel("Value")
for bar, val in zip(bars, sqi_values):
    ax3.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f" {val:.3f}", va="center", fontsize=8)

# ─── Plot 4: HRV Time Domain Bar ───────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
if hrv_time_results:
    hrv_names = list(hrv_time_results.keys())
    hrv_values = list(hrv_time_results.values())
    colors_hrv = plt.cm.plasma(np.linspace(0.2, 0.8, len(hrv_names)))
    bars_hrv = ax4.barh(hrv_names, hrv_values, color=colors_hrv, edgecolor="white")
    ax4.set_title("HRV Time Domain Metrics", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Value")
    for bar, val in zip(bars_hrv, hrv_values):
        ax4.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f" {val:.2f}", va="center", fontsize=8)
else:
    ax4.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("HRV Time Domain Metrics", fontsize=12)

# ─── Plot 5: RR Interval Histogram ─────────────────────────────
ax5 = fig.add_subplot(gs[3, 0])
if len(peaks) > 2:
    ax5.hist(rr_intervals, bins=30, color="teal", edgecolor="white", alpha=0.8)
    ax5.axvline(np.mean(rr_intervals), color="red", linestyle="--",
                label=f"Mean={np.mean(rr_intervals):.0f}ms")
    ax5.set_title("RR Interval Distribution", fontsize=12, fontweight="bold")
    ax5.set_xlabel("RR Interval (ms)")
    ax5.set_ylabel("Count")
    ax5.legend()
else:
    ax5.text(0.5, 0.5, "Insufficient peaks", ha="center", va="center", transform=ax5.transAxes)

# ─── Plot 6: Poincaré Plot ─────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 1])
if len(peaks) > 3:
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    ax6.scatter(rr_n, rr_n1, s=10, alpha=0.6, color="darkorange")
    ax6.plot([min(rr_n), max(rr_n)], [min(rr_n), max(rr_n)],
             "k--", alpha=0.3, label="Identity line")
    ax6.set_title("Poincaré Plot (RRn vs RRn+1)", fontsize=12, fontweight="bold")
    ax6.set_xlabel("RR_n (ms)")
    ax6.set_ylabel("RR_n+1 (ms)")
    ax6.legend()
    ax6.set_aspect("equal", adjustable="box")
else:
    ax6.text(0.5, 0.5, "Insufficient peaks", ha="center", va="center", transform=ax6.transAxes)

# ─── Plot 7: Segment SQI Heatmap ───────────────────────────────
ax7 = fig.add_subplot(gs[4, :])
if len(segment_sqis) > 0:
    seg_df = pd.DataFrame(segment_sqis)
    heatmap_cols = ["kurtosis", "skewness", "entropy", "snr", "zcr", "mcr", "perfusion"]
    available_cols = [c for c in heatmap_cols if c in seg_df.columns]
    if len(available_cols) > 0:
        heatmap_data = seg_df[available_cols].values.astype(float).T
        im = ax7.imshow(heatmap_data, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax7.set_yticks(range(len(available_cols)))
        ax7.set_yticklabels(available_cols, fontsize=9)
        ax7.set_xticks(range(len(seg_df)))
        ax7.set_xticklabels([f"Seg {i+1}" for i in range(len(seg_df))], fontsize=9)
        ax7.set_title("Segment-Level SQI Heatmap", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax7, orientation="horizontal", pad=0.15, shrink=0.6)

        # Annotate values
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                ax7.text(j, i, f"{heatmap_data[i, j]:.2f}",
                         ha="center", va="center", fontsize=7, color="black")

# ─── Plot 8: Heart Rate over time ──────────────────────────────
ax8 = fig.add_subplot(gs[5, 0])
if len(peaks) > 2:
    instantaneous_hr = 60000 / rr_intervals  # bpm
    hr_time = np.array(peaks[1:]) / sampling_rate
    ax8.plot(hr_time, instantaneous_hr, color="crimson", linewidth=1.0)
    ax8.fill_between(hr_time, instantaneous_hr, alpha=0.2, color="crimson")
    ax8.axhline(np.mean(instantaneous_hr), color="gray", linestyle="--",
                alpha=0.5, label=f"Mean HR={np.mean(instantaneous_hr):.0f} bpm")
    ax8.set_title("Instantaneous Heart Rate", fontsize=12, fontweight="bold")
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Heart Rate (bpm)")
    ax8.legend()
    ax8.grid(True, alpha=0.3)
else:
    ax8.text(0.5, 0.5, "Insufficient peaks", ha="center", va="center", transform=ax8.transAxes)

# ─── Plot 9: HR Statistics Summary ─────────────────────────────
ax9 = fig.add_subplot(gs[5, 1])
if hr_results:
    hr_stat_names = list(hr_results.keys())
    hr_stat_values = list(hr_results.values())
    colors_hr = plt.cm.magma(np.linspace(0.2, 0.8, len(hr_stat_names)))
    bars_hr = ax9.barh(hr_stat_names, hr_stat_values, color=colors_hr, edgecolor="white")
    ax9.set_title("Heart Rate Statistics", fontsize=12, fontweight="bold")
    ax9.set_xlabel("Value")
    for bar, val in zip(bars_hr, hr_stat_values):
        ax9.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f" {val:.1f}", va="center", fontsize=8)
else:
    ax9.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax9.transAxes)

plt.suptitle("PPG Signal Quality Analysis Report", fontsize=18, fontweight="bold", y=0.995)
plt.savefig(os.path.join(os.path.dirname(file_in), "..", "ppg_quality_check", "ppg_analysis_report.png"),
            dpi=150, bbox_inches="tight")
print("  Report saved to: ppg_analysis_report.png")
plt.show()

# ════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
total_metrics = len(standard_results) + len(hrv_time_results) + len(hr_results) + len(hrv_freq_results) + len(dtw_results) + len(rpeak_results) + len(waveform_results)
try:
    total_metrics += len(poincare)
except NameError:
    pass
print(f"  Total metrics computed: {total_metrics}")
print(f"  Segments analyzed    : {n_segments}")
print(f"  Signal quality       : ", end="")

# Simple quality assessment based on key metrics
if standard_results.get("SNR", 0) > 5 and standard_results.get("Kurtosis", 0) > 0:
    print("GOOD ✓")
elif standard_results.get("SNR", 0) > 2:
    print("FAIR ⚠")
else:
    print("POOR ✗")
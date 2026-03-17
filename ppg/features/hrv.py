"""
HRV Feature Extraction
=======================
Time-domain, frequency-domain, Poincaré, and full HRV analysis
(via vital_sqi / hrvanalysis).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from vital_sqi.sqi.hrv_sqi import (
    # Time domain
    cvnn_sqi,
    cvsd_sqi,
    mean_nn_sqi,
    median_nn_sqi,
    pnn_sqi,
    rmssd_sqi,
    sdnn_sqi,
    sdsd_sqi,
    # HR statistics
    hr_max_sqi,
    hr_mean_sqi,
    hr_median_sqi,
    hr_min_sqi,
    hr_range_sqi,
    hr_std_sqi,
    # Frequency domain
    absolute_power_sqi,
    lf_hf_ratio_sqi,
    log_power_sqi,
    normalized_power_sqi,
    peak_frequency_sqi,
    relative_power_sqi,
    # Poincaré
    poincare_features_sqi,
    # Full HRV
    get_all_features_hrva,
)

# HRV frequency bands: (display_name, f_min, f_max)
HRV_FREQ_BANDS = [
    ("LF", 0.04, 0.15),
    ("HF", 0.15, 0.40),
]


def compute_hrv_time(nn_intervals: np.ndarray) -> Dict[str, float]:
    """Time-domain HRV metrics. Returns {} if not enough intervals."""
    if len(nn_intervals) < 3:
        return {}
    return {
        "Mean NN (ms)": mean_nn_sqi(nn_intervals),
        "Median NN (ms)": median_nn_sqi(nn_intervals),
        "SDNN (ms)": sdnn_sqi(nn_intervals),
        "SDSD (ms)": sdsd_sqi(nn_intervals),
        "RMSSD (ms)": rmssd_sqi(nn_intervals),
        "CVSD": cvsd_sqi(nn_intervals),
        "CVNN": cvnn_sqi(nn_intervals),
        "pNN50 (%)": pnn_sqi(nn_intervals, exceed=50),
        "pNN20 (%)": pnn_sqi(nn_intervals, exceed=20),
    }


def compute_hr_stats(nn_intervals: np.ndarray) -> Dict[str, float]:
    """Heart rate summary statistics derived from NN intervals."""
    if len(nn_intervals) < 3:
        return {}
    return {
        "HR Mean (bpm)": hr_mean_sqi(nn_intervals),
        "HR Median (bpm)": hr_median_sqi(nn_intervals),
        "HR Min (bpm)": hr_min_sqi(nn_intervals),
        "HR Max (bpm)": hr_max_sqi(nn_intervals),
        "HR Std (bpm)": hr_std_sqi(nn_intervals),
        "HR Out-of-Range (%, 40-200)": hr_range_sqi(nn_intervals),
    }


def compute_hrv_freq(nn_intervals: np.ndarray) -> Dict[str, float]:
    """
    Frequency-domain HRV metrics for LF and HF bands.

    Requires at least 10 NN intervals; returns {} otherwise.
    Silently skips individual metrics that throw exceptions.
    """
    if len(nn_intervals) < 10:
        return {}

    results: Dict[str, float] = {}
    try:
        for band_name, f_min, f_max in HRV_FREQ_BANDS:
            try:
                results[f"Peak Frequency ({band_name})"] = peak_frequency_sqi(
                    nn_intervals, f_min=f_min, f_max=f_max
                )
            except Exception:
                pass
            try:
                results[f"Absolute Power ({band_name})"] = absolute_power_sqi(
                    nn_intervals, f_min=f_min, f_max=f_max
                )
            except Exception:
                pass
            try:
                results[f"Log Power ({band_name})"] = log_power_sqi(
                    nn_intervals, f_min=f_min, f_max=f_max
                )
            except Exception:
                pass
            try:
                results[f"Relative Power ({band_name})"] = relative_power_sqi(
                    nn_intervals, f_min=f_min, f_max=f_max
                )
            except Exception:
                pass
        try:
            results["Normalized Power"] = normalized_power_sqi(nn_intervals)
        except Exception:
            pass
        try:
            results["LF/HF Ratio"] = lf_hf_ratio_sqi(nn_intervals)
        except Exception:
            pass
    except Exception as e:
        results["_error"] = str(e)

    return results


def compute_poincare(nn_intervals: np.ndarray) -> Dict[str, float]:
    """Poincaré plot features (SD1, SD2, area, ratio)."""
    if len(nn_intervals) < 4:
        return {}
    try:
        return dict(poincare_features_sqi(nn_intervals))
    except Exception:
        return {}


def compute_hrv_full(
    processed: np.ndarray, sampling_rate: int
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Full HRV analysis via hrvanalysis (time, freq, geometric, CSI/CVI).

    Returns four dicts: (time_feats, freq_feats, geom_feats, csi_cvi_feats).
    Any empty dict means that group was unavailable / threw an error.
    """
    try:
        return get_all_features_hrva(
            processed,
            sample_rate=sampling_rate,
            rpeak_method=0,
            wave_type="ppg",
        )
    except Exception:
        return {}, {}, {}, {}

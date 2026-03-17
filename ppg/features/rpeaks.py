"""
R-peak Based SQI
=================
Ectopic beat ratio, autocorrelogram peaks, and multi-detector MSQ.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from vital_sqi.sqi.rpeaks_sqi import correlogram_sqi, ectopic_sqi, msq_sqi


def compute_rpeak_sqi(
    processed: np.ndarray,
    sampling_rate: int,
    ectopic_rule_index: int = 1,
    rpeak_detector: int = 0,
    corr_time_lag: int = 3,
    corr_n_selection: int = 3,
    msq_detector_1: int = 7,
    msq_detector_2: int = 6,
) -> Dict[str, Any]:
    """
    Compute all R-peak–based SQI metrics.

    Parameters
    ----------
    processed        : smoothed PPG signal
    sampling_rate    : in Hz
    ectopic_rule_index : Malik(1) or Karlsson(2) ectopic rule
    rpeak_detector   : detector index for ectopic_sqi
    corr_time_lag    : seconds of lag for correlogram
    corr_n_selection : number of peaks to select from correlogram
    msq_detector_1/2 : two detector indices for MSQ consistency

    Returns
    -------
    dict[metric_name, value_or_array]
    Errors are stored as float("nan") so callers always get a numeric value.
    """
    results: Dict[str, Any] = {}

    # ── Ectopic ratio ─────────────────────────────────────────────────────
    try:
        results["Ectopic Ratio (Malik)"] = ectopic_sqi(
            processed,
            rule_index=ectopic_rule_index,
            sample_rate=sampling_rate,
            rpeak_detector=rpeak_detector,
            wave_type="ppg",
        )
    except Exception:
        results["Ectopic Ratio (Malik)"] = float("nan")

    # ── Correlogram ───────────────────────────────────────────────────────
    try:
        corr_features = correlogram_sqi(
            processed,
            sample_rate=sampling_rate,
            time_lag=corr_time_lag,
            n_selection=corr_n_selection,
        )
        results["Correlogram"] = corr_features
    except Exception:
        results["Correlogram"] = np.array([])

    # ── Multi-detector MSQ ────────────────────────────────────────────────
    try:
        results["MSQ (Multi-Detector)"] = msq_sqi(
            processed,
            peak_detector_1=msq_detector_1,
            peak_detector_2=msq_detector_2,
            wave_type="ppg",
        )
    except Exception:
        results["MSQ (Multi-Detector)"] = float("nan")

    return results

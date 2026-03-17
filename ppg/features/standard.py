"""
Standard SQI Metrics
=====================
Statistical signal quality indicators computed on the processed PPG.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from vital_sqi.sqi.standard_sqi import (
    entropy_sqi,
    kurtosis_sqi,
    mean_crossing_rate_sqi,
    perfusion_sqi,
    signal_to_noise_sqi,
    skewness_sqi,
    zero_crossings_rate_sqi,
)


def compute_standard_sqi(
    processed: np.ndarray,
    raw: np.ndarray,
) -> Dict[str, float]:
    """
    Compute the seven standard SQI metrics.

    Parameters
    ----------
    processed : filtered + smoothed signal
    raw       : resampled but unfiltered signal (needed for SNR / perfusion)

    Returns
    -------
    dict[metric_name, value]
    """
    return {
        "Kurtosis": kurtosis_sqi(processed),
        "Skewness": skewness_sqi(processed),
        "Entropy": entropy_sqi(processed),
        "SNR": signal_to_noise_sqi(raw),
        "Perfusion Index (%)": perfusion_sqi(raw, processed),
        "Zero Crossing Rate": zero_crossings_rate_sqi(processed),
        "Mean Crossing Rate": mean_crossing_rate_sqi(processed),
    }

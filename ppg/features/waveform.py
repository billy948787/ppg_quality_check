"""
Waveform SQI: DTW template matching, band energy, beat template correlation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import resample as scipy_resample

from vital_sqi.sqi.dtw_sqi import dtw_sqi
from vital_sqi.sqi.waveform_sqi import band_energy_sqi, lf_energy_sqi

from ..config import BandDef

# ── DTW template registry ─────────────────────────────────────────────────────
DTW_TEMPLATES: Dict[int, str] = {
    0: "Nonlinear Dynamic System",
    1: "Dual Double Frequency",
    2: "Absolute Dual Skewness",
}


def compute_dtw(
    processed: np.ndarray,
    valleys: np.ndarray,
    peaks: np.ndarray,
    template_types: Tuple[int, ...] = (0, 1, 2),
    template_size: int = 100,
    n_per_beat: int = 10,
) -> Dict[str, float]:
    """
    DTW template matching SQI.

    Computes:
      - Single representative beat vs. each template type
      - Per-beat average DTW (template 0) across up to `n_per_beat` beats

    Parameters
    ----------
    processed      : smoothed PPG signal
    valleys        : valley indices (fiducial points)
    peaks          : peak indices
    template_types : which DTW template IDs to evaluate
    template_size  : number of samples in the reference template
    n_per_beat     : max beats for per-beat DTW average

    Returns
    -------
    dict[metric_name, value]
    """
    results: Dict[str, float] = {}

    if len(peaks) < 3 or len(valleys) < 2:
        return results

    # ── Single representative beat ────────────────────────────────────────
    mid_idx = len(peaks) // 2
    if 0 < mid_idx < len(peaks) - 1:
        b_start = valleys[min(mid_idx - 1, len(valleys) - 1)]
        b_end = valleys[min(mid_idx, len(valleys) - 1)]
        beat = processed[b_start:b_end]

        if len(beat) > 10:
            for t in template_types:
                name = DTW_TEMPLATES.get(t, f"Template_{t}")
                try:
                    results[f"DTW {name}"] = dtw_sqi(
                        beat, template_type=t, template_size=template_size
                    )
                except Exception as e:
                    results[f"DTW {name} (error)"] = float("nan")

    # ── Per-beat average (template 0) ─────────────────────────────────────
    if len(peaks) >= 5 and len(valleys) >= 4:
        scores: List[float] = []
        for i in range(min(n_per_beat, len(valleys) - 1)):
            try:
                seg = processed[valleys[i] : valleys[i + 1]]
                if len(seg) > 10:
                    scores.append(
                        dtw_sqi(seg, template_type=0, template_size=template_size)
                    )
            except Exception:
                pass
        if scores:
            results["DTW PerBeat Mean"] = float(np.mean(scores))
            results["DTW PerBeat Std"] = float(np.std(scores))

    return results


def compute_waveform_energy(
    processed: np.ndarray,
    sampling_rate: int,
    bands: Optional[List[BandDef]] = None,
) -> Dict[str, float]:
    """
    Band energy SQI for a list of frequency bands.

    Parameters
    ----------
    processed     : smoothed PPG signal
    sampling_rate : in Hz
    bands         : list of BandDef.  If None, uses the four default bands.

    Returns
    -------
    dict[f"Energy {band.name}", value]
    """
    from ..config import DEFAULT_WAVEFORM_BANDS

    if bands is None:
        bands = DEFAULT_WAVEFORM_BANDS

    results: Dict[str, float] = {}
    for band in bands:
        lo, hi = band.low, band.high
        key = f"Energy {band.name}"
        try:
            if lo is None and hi is None:
                results[key] = band_energy_sqi(
                    processed, sampling_rate=sampling_rate, band=None
                )
            elif lo == 0.0 or lo is None:
                # Use lf_energy_sqi for bands starting at/near DC
                results[key] = lf_energy_sqi(
                    processed,
                    sampling_rate=sampling_rate,
                    band=[lo or 0.0, hi],
                )
            else:
                results[key] = band_energy_sqi(
                    processed, sampling_rate=sampling_rate, band=[lo, hi]
                )
        except Exception:
            results[key] = float("nan")

    return results


def compute_beat_template_corr(
    processed: np.ndarray,
    valleys: np.ndarray,
    max_beats: int = 30,
) -> Tuple[float, int]:
    """
    Average Pearson correlation between each beat and the mean beat template.

    High (→1): consistent waveform shape.
    Low values indicate morphology variability or artefacts.

    Returns
    -------
    (mean_correlation, n_beats_used)
    """
    if len(valleys) < 4:
        return float("nan"), 0

    beat_len = int(np.median(np.diff(valleys)))
    if beat_len <= 10:
        return float("nan"), 0

    templates: List[np.ndarray] = []
    for i in range(min(max_beats, len(valleys) - 1)):
        b = processed[valleys[i] : valleys[i + 1]]
        if len(b) > 10:
            b_rs = scipy_resample(b, beat_len)
            b_n = (b_rs - b_rs.mean()) / (b_rs.std() + 1e-8)
            templates.append(b_n)

    n_beats = len(templates)
    if n_beats < 2:
        return float("nan"), n_beats

    avg = np.mean(templates, axis=0)
    corrs = [np.corrcoef(t, avg)[0, 1] for t in templates]
    corrs_ok = [c for c in corrs if not np.isnan(c)]
    return (float(np.mean(corrs_ok)) if corrs_ok else float("nan")), n_beats

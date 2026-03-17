"""
PPG Data Loader
===============
Flexible CSV loading that:
  - Auto-detects signal columns from known patterns (or uses config override)
  - Auto-detects timestamp column from known names (or uses config override)
  - Handles numeric timestamps in ns / µs / ms / s, and string datetimes
  - Resamples every channel to a uniform time grid
  - Returns one SignalData per channel, enabling multi-band analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PPGConfig, SIGNAL_COLUMN_PATTERNS, TIMESTAMP_COLUMN_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Public data container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SignalData:
    """Loaded and uniformly resampled data for one PPG channel."""

    channel_name: str
    raw_signal: np.ndarray  # original samples (before resampling)
    signal: np.ndarray  # resampled to uniform grid
    time: np.ndarray  # uniform time axis in seconds (starts at 0)
    sampling_rate: int  # Hz after resampling

    # ── Metadata ──────────────────────────────────────────────────────────
    duration_s: float
    n_original: int
    n_duplicate_timestamps: int
    duplicate_ratio: float  # fraction of diffs that were zero

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SignalData(channel={self.channel_name!r}, "
            f"samples={len(self.signal)}, "
            f"duration={self.duration_s:.1f}s, "
            f"fs={self.sampling_rate}Hz)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _detect_signal_columns(df: pd.DataFrame, patterns: List[str]) -> List[str]:
    """Return column names that match any pattern (case-insensitive substring)."""
    col_lower = {c: c.lower() for c in df.columns}
    seen: set = set()
    found: List[str] = []
    for pat in patterns:
        for orig, low in col_lower.items():
            if pat in low and orig not in seen:
                found.append(orig)
                seen.add(orig)
    return found


def _detect_timestamp_column(df: pd.DataFrame, known_names: List[str]) -> Optional[str]:
    """Return the first column whose lower-cased name matches a known timestamp name."""
    col_lower = {c.lower(): c for c in df.columns}
    for name in known_names:
        if name in col_lower:
            return col_lower[name]
    return None


def _parse_timestamps_to_relative_ms(time_col: pd.Series) -> np.ndarray:
    """
    Convert any timestamp format to a relative millisecond array starting at 0.

    Supports:
      - Numeric nanoseconds  (~1e18)
      - Numeric microseconds (~1e15)
      - Numeric milliseconds (~1e12)
      - Numeric seconds       (<1e10)
      - String / datetime-parseable values
    """
    if np.issubdtype(time_col.dtype, np.number):
        ts = time_col.values.astype(float)
        mean_ts = float(np.nanmean(ts))

        if mean_ts > 1e17:
            ts_ms = ts / 1e6  # ns → ms
        elif mean_ts > 1e14:
            ts_ms = ts / 1_000.0  # µs → ms
        elif mean_ts < 1e10:
            ts_ms = ts * 1_000.0  # s  → ms
        else:
            ts_ms = ts  # already ms
    else:
        dt = pd.to_datetime(time_col)
        ts_ms = (dt - dt.iloc[0]).dt.total_seconds().values * 1_000.0

    # Make relative (start at 0)
    return ts_ms - ts_ms[0]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def load_ppg_data(config: PPGConfig) -> Dict[str, SignalData]:
    """
    Load PPG data from a CSV file and return one SignalData per channel.

    Column resolution order (signal)
    ---------------------------------
    1. config.signal_columns  – explicit list → used as-is
    2. Auto-detection via SIGNAL_COLUMN_PATTERNS

    Column resolution order (timestamp)
    -------------------------------------
    1. config.timestamp_column – explicit name → used as-is
    2. Auto-detection via TIMESTAMP_COLUMN_NAMES

    Returns
    -------
    dict[channel_name, SignalData]  – one entry per signal column found
    """
    df = pd.read_csv(config.file_path)
    df.columns = df.columns.str.strip()

    # ── Resolve signal columns ────────────────────────────────────────────
    if config.signal_columns:
        sig_cols = config.signal_columns
        missing = [c for c in sig_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Configured signal columns not found in CSV: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )
    else:
        sig_cols = _detect_signal_columns(df, SIGNAL_COLUMN_PATTERNS)
        if not sig_cols:
            raise ValueError(
                "No signal columns auto-detected.\n"
                f"Patterns tried: {SIGNAL_COLUMN_PATTERNS}\n"
                f"Available columns: {list(df.columns)}\n"
                "Set PPGConfig.signal_columns explicitly to override."
            )

    # ── Resolve timestamp column ──────────────────────────────────────────
    if config.timestamp_column:
        ts_col_name = config.timestamp_column
        if ts_col_name not in df.columns:
            raise ValueError(
                f"Configured timestamp column '{ts_col_name}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )
    else:
        ts_col_name = _detect_timestamp_column(df, TIMESTAMP_COLUMN_NAMES)
        if ts_col_name is None:
            raise ValueError(
                "No timestamp column auto-detected.\n"
                f"Names tried: {TIMESTAMP_COLUMN_NAMES}\n"
                f"Available columns: {list(df.columns)}\n"
                "Set PPGConfig.timestamp_column explicitly to override."
            )

    # ── Parse timestamps ──────────────────────────────────────────────────
    ts_ms = _parse_timestamps_to_relative_ms(df[ts_col_name])
    ts_diffs = np.diff(ts_ms)
    duration_s = float(ts_ms[-1]) / 1_000.0
    n_duplicates = int(np.sum(ts_diffs == 0))
    dup_ratio = n_duplicates / max(len(ts_diffs), 1)

    # ── Build uniform time grid ───────────────────────────────────────────
    sr = config.target_sampling_rate
    t_raw_s = ts_ms / 1_000.0
    t_uniform = np.arange(0.0, duration_s, 1.0 / sr)

    # ── Build one SignalData per channel ──────────────────────────────────
    result: Dict[str, SignalData] = {}
    for col in sig_cols:
        raw = df[col].values.astype(float)
        resampled = np.interp(t_uniform, t_raw_s, raw)

        result[col] = SignalData(
            channel_name=col,
            raw_signal=raw,
            signal=resampled,
            time=t_uniform,
            sampling_rate=sr,
            duration_s=duration_s,
            n_original=len(raw),
            n_duplicate_timestamps=n_duplicates,
            duplicate_ratio=dup_ratio,
        )

    return result


def print_load_summary(channels: Dict[str, SignalData]) -> None:
    """Pretty-print a summary of the loaded channels."""
    first = next(iter(channels.values()))
    print("=" * 70)
    print(f"Loaded {len(channels)} channel(s) from CSV")
    print("=" * 70)
    print(f"  Duration          : {first.duration_s:.1f} s")
    print(f"  Sampling rate     : {first.sampling_rate} Hz (uniform grid)")
    print(
        f"  Duplicate timestamps: {first.n_duplicate_timestamps} "
        f"({100 * first.duplicate_ratio:.1f}%)"
    )
    print()
    for name, sd in channels.items():
        print(
            f"  [{name}]  "
            f"original={sd.n_original} samples  "
            f"resampled={len(sd.signal)} samples  "
            f"range=[{sd.signal.min():.0f}, {sd.signal.max():.0f}]"
        )

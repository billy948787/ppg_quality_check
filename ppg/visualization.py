"""
Visualization
==============
All matplotlib plotting for:
  - Single-channel analysis report (preprocessing + main analysis)
  - Multi-band comparison dashboard (when > 1 channel is analysed)
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PPGConfig
from .pipeline import ChannelResult


# ─────────────────────────────────────────────────────────────────────────────
# Colour scheme helpers
# ─────────────────────────────────────────────────────────────────────────────

_CHANNEL_COLORS = [
    "steelblue",
    "crimson",
    "forestgreen",
    "darkorange",
    "mediumpurple",
    "saddlebrown",
    "deeppink",
    "teal",
]

_GRADE_ZONES = [
    (0, 40, "#d32f2f", "Poor"),
    (40, 55, "#f57c00", "Marginal"),
    (55, 70, "#fbc02d", "Fair"),
    (70, 85, "#689f38", "Good"),
    (85, 100, "#2e7d32", "Excellent"),
]


def _channel_color(idx: int) -> str:
    return _CHANNEL_COLORS[idx % len(_CHANNEL_COLORS)]


def _savefig(fig: plt.Figure, filename: str, config: PPGConfig) -> None:
    if config.save_plots:
        path = os.path.join(config.output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
    if not config.show_plots:
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing verification plot (one channel)
# ─────────────────────────────────────────────────────────────────────────────


def plot_preprocessing(result: ChannelResult, config: PPGConfig) -> None:
    """Three-panel preprocessing verification figure."""
    pre = result.preprocessed
    time = pre.time
    ch = result.channel_name

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle(
        f"PPG Preprocessing Verification  [{ch}]", fontsize=16, fontweight="bold"
    )

    axes[0].plot(time, pre.raw, linewidth=0.6, color="steelblue")
    axes[0].set_title(
        f"(a) Raw (resampled)  range=[{pre.raw.min():.0f}, {pre.raw.max():.0f}]"
    )
    axes[0].set_ylabel("ADC Value")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, pre.filtered, linewidth=0.6, color="crimson")
    axes[1].set_title(f"(b) After Bandpass Filter ({pre.lowcut}–{pre.highcut} Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        time, pre.filtered, linewidth=0.6, alpha=0.4, color="crimson", label="Filtered"
    )
    axes[2].plot(
        time, pre.smoothed, linewidth=1.0, color="forestgreen", label="Smoothed (used)"
    )
    axes[2].set_title("(c) Filtered vs Smoothed")
    axes[2].set_ylabel("Amplitude")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _savefig(fig, f"ppg_preprocessing_{ch}.png", config)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis report (one channel, 11 panels)
# ─────────────────────────────────────────────────────────────────────────────


def plot_analysis_report(result: ChannelResult, config: PPGConfig) -> None:
    """Comprehensive 11-panel analysis report for one channel."""
    pre = result.preprocessed
    pk = result.peaks
    ch = result.channel_name
    time = pre.time
    sr = pre.sampling_rate
    rr = pk.rr_intervals
    c = result.composite

    fig = plt.figure(figsize=(20, 28))
    gs = gridspec.GridSpec(7, 2, hspace=0.45, wspace=0.3)

    # ── 1. Raw vs Filtered ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, pre.raw, alpha=0.5, linewidth=0.5, label="Raw", color="steelblue")
    ax1.plot(time, pre.processed, linewidth=1.0, label="Filtered", color="crimson")
    ax1.set_title(f"Raw vs Filtered PPG  [{ch}]", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── 2. Peaks on filtered signal ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time, pre.processed, linewidth=0.8, color="steelblue", label="Filtered")
    if len(pk.valleys) > 0:
        ax2.scatter(
            pk.valleys / sr,
            pre.processed[pk.valleys],
            color="red",
            s=20,
            zorder=5,
            label=f"Valleys ({len(pk.valleys)})",
        )
    if len(pk.peaks) > 0:
        ax2.scatter(
            pk.peaks / sr,
            pre.processed[pk.peaks],
            color="green",
            s=15,
            zorder=5,
            label=f"Peaks ({len(pk.peaks)})",
        )
    ax2.set_title("Peak Detection Results", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # ── 3. Standard SQI bar ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    if result.standard_sqi:
        names = list(result.standard_sqi.keys())
        vals = list(result.standard_sqi.values())
        cols = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        bars = ax3.barh(names, vals, color=cols, edgecolor="white")
        ax3.set_title("Standard SQI Metrics", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Value")
        for bar, val in zip(bars, vals):
            ax3.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {val:.3f}",
                va="center",
                fontsize=8,
            )

    # ── 4. HRV time domain ───────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    if result.hrv_time:
        names = list(result.hrv_time.keys())
        vals = list(result.hrv_time.values())
        cols = plt.cm.plasma(np.linspace(0.2, 0.8, len(names)))
        bars = ax4.barh(names, vals, color=cols, edgecolor="white")
        ax4.set_title("HRV Time Domain Metrics", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Value")
        for bar, val in zip(bars, vals):
            ax4.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {val:.2f}",
                va="center",
                fontsize=8,
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "Insufficient data",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("HRV Time Domain Metrics", fontsize=12)

    # ── 5. RR histogram ──────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    if len(rr) > 2:
        ax5.hist(rr, bins=30, color="teal", edgecolor="white", alpha=0.8)
        ax5.axvline(
            np.mean(rr), color="red", linestyle="--", label=f"Mean={np.mean(rr):.0f} ms"
        )
        ax5.set_title("RR Interval Distribution", fontsize=12, fontweight="bold")
        ax5.set_xlabel("RR (ms)")
        ax5.set_ylabel("Count")
        ax5.legend()
    else:
        ax5.text(
            0.5,
            0.5,
            "Insufficient peaks",
            ha="center",
            va="center",
            transform=ax5.transAxes,
        )

    # ── 6. Poincaré ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    if len(rr) > 3:
        ax6.scatter(rr[:-1], rr[1:], s=10, alpha=0.6, color="darkorange")
        lo, hi = rr.min(), rr.max()
        ax6.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="Identity")
        ax6.set_title("Poincaré Plot (RRn vs RRn+1)", fontsize=12, fontweight="bold")
        ax6.set_xlabel("RR_n (ms)")
        ax6.set_ylabel("RR_n+1 (ms)")
        ax6.legend()
        ax6.set_aspect("equal", adjustable="box")
    else:
        ax6.text(
            0.5,
            0.5,
            "Insufficient peaks",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )

    # ── 7. Segment SQI heatmap ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    if result.segments:
        seg_df = pd.DataFrame(
            [
                {
                    "kurtosis": s.kurtosis,
                    "skewness": s.skewness,
                    "entropy": s.entropy,
                    "snr": s.snr,
                    "zcr": s.zcr,
                    "mcr": s.mcr,
                    "perfusion": s.perfusion,
                }
                for s in result.segments
            ]
        )
        heatmap = seg_df.values.T
        im = ax7.imshow(heatmap, aspect="auto", cmap="RdYlGn", interpolation="nearest")
        ax7.set_yticks(range(len(seg_df.columns)))
        ax7.set_yticklabels(list(seg_df.columns), fontsize=9)
        ax7.set_xticks(range(len(result.segments)))
        ax7.set_xticklabels(
            [f"Seg {s.segment_idx}" for s in result.segments], fontsize=9
        )
        ax7.set_title("Segment-Level SQI Heatmap", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax7, orientation="horizontal", pad=0.15, shrink=0.6)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                ax7.text(
                    j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center", fontsize=7
                )

    # ── 8. Instantaneous HR ──────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[5, 0])
    if len(rr) > 2:
        inst_hr = 60_000 / rr
        hr_time = pk.valleys[1:] / sr
        ax8.plot(hr_time, inst_hr, color="crimson", linewidth=1.0)
        ax8.fill_between(hr_time, inst_hr, alpha=0.2, color="crimson")
        ax8.axhline(
            np.mean(inst_hr),
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Mean={np.mean(inst_hr):.0f} bpm",
        )
        ax8.set_title("Instantaneous Heart Rate", fontsize=12, fontweight="bold")
        ax8.set_xlabel("Time (s)")
        ax8.set_ylabel("HR (bpm)")
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(
            0.5,
            0.5,
            "Insufficient peaks",
            ha="center",
            va="center",
            transform=ax8.transAxes,
        )

    # ── 9. HR stats ──────────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[5, 1])
    if result.hr_stats:
        names = list(result.hr_stats.keys())
        vals = list(result.hr_stats.values())
        cols = plt.cm.magma(np.linspace(0.2, 0.8, len(names)))
        bars = ax9.barh(names, vals, color=cols, edgecolor="white")
        ax9.set_title("Heart Rate Statistics", fontsize=12, fontweight="bold")
        ax9.set_xlabel("Value")
        for bar, val in zip(bars, vals):
            ax9.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {val:.1f}",
                va="center",
                fontsize=8,
            )

    # ── 10. Composite score gauge ─────────────────────────────────────────
    ax10 = fig.add_subplot(gs[6, 0])
    for lo, hi, col, lbl in _GRADE_ZONES:
        ax10.barh(
            0, hi - lo, left=lo, height=0.5, color=col, alpha=0.25, edgecolor="none"
        )
        ax10.text(
            (lo + hi) / 2,
            0.45,
            lbl,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="dimgray",
        )
    score = c.score if c else 0.0
    bar_col = next(col for lo, hi, col, _ in _GRADE_ZONES if lo <= score < hi + 0.001)
    ax10.barh(0, score, height=0.5, color=bar_col, alpha=0.9, zorder=3)
    ax10.axvline(
        score, color="black", linewidth=2.0, linestyle="--", alpha=0.75, zorder=4
    )
    ax10.text(
        score,
        -0.42,
        f"{score:.1f}",
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=bar_col,
    )
    ax10.set_xlim(0, 100)
    ax10.set_ylim(-0.6, 0.9)
    ax10.set_yticks([])
    ax10.set_xlabel("Score (0 – 100)")
    grade_str = f"{c.grade} – {c.label}" if c else "N/A"
    ax10.set_title(
        f"Composite SQI: {score:.1f}/100  |  {grade_str}",
        fontsize=12,
        fontweight="bold",
    )
    ax10.grid(True, axis="x", alpha=0.3)

    # ── 11. Sub-score bars ────────────────────────────────────────────────
    ax11 = fig.add_subplot(gs[6, 1])
    if c:
        sc_names = [n for n, _, s in c.sub_scores if not np.isnan(s)]
        sc_vals = [s for _, _, s in c.sub_scores if not np.isnan(s)]
        sc_cols = [
            "#2e7d32" if s >= 70 else "#f57c00" if s >= 40 else "#d32f2f"
            for s in sc_vals
        ]
        y_pos = list(range(len(sc_names)))
        bars11 = ax11.barh(y_pos, sc_vals, color=sc_cols, edgecolor="white", alpha=0.85)
        ax11.axvline(
            70,
            color="green",
            linewidth=1.2,
            linestyle="--",
            alpha=0.6,
            label="Good ≥70",
        )
        ax11.axvline(
            40,
            color="orange",
            linewidth=1.2,
            linestyle="--",
            alpha=0.6,
            label="Fair ≥40",
        )
        ax11.set_yticks(y_pos)
        ax11.set_yticklabels(sc_names, fontsize=9)
        ax11.set_xlim(0, 108)
        ax11.set_xlabel("Sub-Score (0 – 100)")
        ax11.set_title("Metric Sub-Scores", fontsize=12, fontweight="bold")
        ax11.legend(fontsize=7.5, loc="lower right")
        for bar, val in zip(bars11, sc_vals):
            ax11.text(
                bar.get_width() + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}",
                va="center",
                fontsize=8.5,
            )

    plt.suptitle(
        f"PPG Signal Quality Analysis Report  [{ch}]",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    _savefig(fig, f"ppg_analysis_report_{ch}.png", config)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-band comparison dashboard
# ─────────────────────────────────────────────────────────────────────────────


def plot_multiband_comparison(
    results: Dict[str, ChannelResult],
    config: PPGConfig,
) -> None:
    """
    Side-by-side comparison of key metrics across all analysed channels/bands.

    Panels
    ------
    1. Processed signals (time domain overlay)
    2. Composite SQI scores (bar)
    3. Standard SQI comparison (grouped bars)
    4. Spectral SNR, Beat Regularity, Template Corr, Clipping Rate
    5. Segment composite score heatmap (channel × segment)
    6. Per-channel quality label summary
    """
    channels = list(results.keys())
    n_ch = len(channels)
    if n_ch < 2:
        return  # nothing to compare

    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.35)

    # ── 1. Processed signal overlay ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for i, (ch, r) in enumerate(results.items()):
        color = _channel_color(i)
        sig = r.preprocessed.processed
        # Normalise to [-1, 1] for visual comparison
        s_n = (sig - sig.mean()) / (sig.std() + 1e-8)
        ax1.plot(
            r.preprocessed.time, s_n + i * 2.5, linewidth=0.7, color=color, label=ch
        )
    ax1.set_title(
        "Processed PPG Signals (normalised, offset per channel)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Normalised Amplitude")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Composite score bar ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    scores = [r.composite.score if r.composite else 0.0 for r in results.values()]
    bar_cols = [
        next(col for lo, hi, col, _ in _GRADE_ZONES if lo <= s < hi + 0.001)
        for s in scores
    ]
    bars = ax2.bar(channels, scores, color=bar_cols, edgecolor="white", alpha=0.85)
    ax2.axhline(70, color="green", linestyle="--", alpha=0.6, label="Good ≥70")
    ax2.axhline(40, color="orange", linestyle="--", alpha=0.6, label="Fair ≥40")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Composite SQI (0–100)")
    ax2.set_title("Composite SQI Score per Channel", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8)
    for bar, score in zip(bars, scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{score:.1f}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # ── 3. Standard SQI grouped bars ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    std_keys = ["Kurtosis", "Skewness", "Entropy", "SNR", "Perfusion Index (%)"]
    x = np.arange(len(std_keys))
    width = 0.8 / n_ch
    for i, (ch, r) in enumerate(results.items()):
        vals = [r.standard_sqi.get(k, float("nan")) for k in std_keys]
        # Clip to finite for display
        vals_disp = [v if np.isfinite(v) else 0.0 for v in vals]
        ax3.bar(
            x + i * width,
            vals_disp,
            width,
            label=ch,
            color=_channel_color(i),
            alpha=0.8,
            edgecolor="white",
        )
    ax3.set_xticks(x + width * (n_ch - 1) / 2)
    ax3.set_xticklabels(
        ["Kurtosis", "Skewness", "Entropy", "SNR", "Perfusion"], rotation=20, ha="right"
    )
    ax3.set_title("Standard SQI Comparison", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, axis="y", alpha=0.3)

    # ── 4. Signal-level quality metrics ──────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    sig_metrics = ["Spectral SNR", "Beat Regularity", "Template Corr", "Clipping Rate"]
    x4 = np.arange(len(sig_metrics))
    width4 = 0.8 / n_ch
    for i, (ch, r) in enumerate(results.items()):
        vals = [
            r.spectral_snr if np.isfinite(r.spectral_snr) else 0.0,
            r.beat_regularity if np.isfinite(r.beat_regularity) else 0.0,
            r.beat_template_corr if np.isfinite(r.beat_template_corr) else 0.0,
            r.clipping_rate if np.isfinite(r.clipping_rate) else 0.0,
        ]
        ax4.bar(
            x4 + i * width4,
            vals,
            width4,
            label=ch,
            color=_channel_color(i),
            alpha=0.8,
            edgecolor="white",
        )
    ax4.set_xticks(x4 + width4 * (n_ch - 1) / 2)
    ax4.set_xticklabels(sig_metrics, rotation=15, ha="right")
    ax4.set_title("Signal-Level Quality Metrics", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8)
    ax4.grid(True, axis="y", alpha=0.3)

    # ── 5. Segment composite score heatmap ────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    max_segs = max(len(r.segments) for r in results.values())
    hmap = np.full((n_ch, max_segs), float("nan"))
    for i, r in enumerate(results.values()):
        for j, seg in enumerate(r.segments):
            hmap[i, j] = seg.composite_score
    masked = np.ma.masked_invalid(hmap)
    im5 = ax5.imshow(
        masked, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100, interpolation="nearest"
    )
    ax5.set_yticks(range(n_ch))
    ax5.set_yticklabels(channels, fontsize=9)
    ax5.set_xticks(range(max_segs))
    ax5.set_xticklabels([f"Seg {j + 1}" for j in range(max_segs)], fontsize=9)
    ax5.set_title(
        "Segment Composite Score Heatmap (per channel)", fontsize=12, fontweight="bold"
    )
    plt.colorbar(im5, ax=ax5, orientation="vertical", shrink=0.8, label="Score 0–100")
    for i in range(n_ch):
        for j in range(max_segs):
            v = hmap[i, j]
            if np.isfinite(v):
                ax5.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8)

    # ── 6. Quality label summary ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis("off")
    col_labels = [
        "Channel",
        "Composite SQI",
        "Grade",
        "Rule Label",
        "Spectral SNR",
        "Template Corr",
        "Beat Reg.",
        "Clipping",
    ]
    table_data = []
    for ch, r in results.items():
        c = r.composite
        table_data.append(
            [
                ch,
                f"{c.score:.1f}" if c else "N/A",
                f"{c.grade} – {c.label}" if c else "N/A",
                r.quality_label,
                f"{r.spectral_snr:.3f}" if np.isfinite(r.spectral_snr) else "N/A",
                f"{r.beat_template_corr:.3f}"
                if np.isfinite(r.beat_template_corr)
                else "N/A",
                f"{r.beat_regularity:.3f}" if np.isfinite(r.beat_regularity) else "N/A",
                f"{r.clipping_rate:.4f}",
            ]
        )
    tbl = ax6.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    ax6.set_title("Multi-Band Summary Table", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle(
        "PPG Multi-Band Quality Comparison", fontsize=18, fontweight="bold", y=0.995
    )
    _savefig(fig, "ppg_multiband_comparison.png", config)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────


def generate_all_plots(
    results: Dict[str, ChannelResult],
    config: PPGConfig,
) -> None:
    """
    Generate all plots for all channels plus (if multi-band) the comparison.

    Parameters
    ----------
    results : dict[channel_name, ChannelResult]
    config  : PPGConfig
    """
    for ch, result in results.items():
        print(f"\n  Plotting preprocessing verification [{ch}]...")
        plot_preprocessing(result, config)
        print(f"  Plotting analysis report [{ch}]...")
        plot_analysis_report(result, config)

    if len(results) > 1:
        print("\n  Plotting multi-band comparison dashboard...")
        plot_multiband_comparison(results, config)

    if config.show_plots:
        plt.show()
    plt.close("all")

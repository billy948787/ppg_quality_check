"""
Rule-Based Quality Classification
===================================
Builds vital_sqi Rule / RuleSet objects from the available SQI metrics
and classifies signal quality as GOOD / FAIR / POOR.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from vital_sqi.rule import Rule, RuleSet


# ─────────────────────────────────────────────────────────────────────────────
# Rule factory helpers
# ─────────────────────────────────────────────────────────────────────────────


def _range_rule(name: str, low: float, high: float) -> Rule:
    """Accept values in [low, high); reject everything outside."""
    r = Rule(name)
    r.update_def(
        op_list=["<", ">", ">=", "<"],
        value_list=[low, low, high, high],
        label_list=["reject", "accept", "reject", "accept"],
    )
    return r


def _gt_rule(name: str, threshold: float) -> Rule:
    """Accept values > threshold."""
    r = Rule(name)
    r.update_def(
        op_list=["<", ">"],
        value_list=[threshold, threshold],
        label_list=["reject", "accept"],
    )
    return r


def _lt_rule(name: str, threshold: float) -> Rule:
    """Accept values < threshold."""
    r = Rule(name)
    r.update_def(
        op_list=["<", ">"],
        value_list=[threshold, threshold],
        label_list=["accept", "reject"],
    )
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def build_rulesets(
    rpeak: Dict[str, Any],
    hr: Dict[str, float],
) -> tuple[RuleSet, RuleSet]:
    """
    Build strict and loose rule sets from available metric values.

    Strict  → GOOD  if all rules pass
    Loose   → FAIR  if all rules pass
    Neither → POOR

    Optional rules (MSQ, Ectopic, HR-Range) are added only if data exist.

    Returns
    -------
    (strict_ruleset, loose_ruleset)
    """
    strict: Dict[int, Rule] = {
        1: _range_rule("Skewness", -0.26, 0.87),
        2: _range_rule("Kurtosis", -1.25, 1.17),
        3: _gt_rule("SNR", 2.0),
        4: _gt_rule("Perfusion", 0.5),
        5: _range_rule("Zero-Crossing-Rate", 0.03, 0.07),
        6: _range_rule("Mean-Crossing-Rate", 0.02, 0.07),
    }
    loose: Dict[int, Rule] = {
        1: _range_rule("Skewness", -1.0, 2.0),
        2: _range_rule("Kurtosis", -2.0, 3.0),
        3: _gt_rule("SNR", 0.0),
        4: _gt_rule("Perfusion", 0.05),
        5: _range_rule("Zero-Crossing-Rate", 0.01, 0.15),
        6: _range_rule("Mean-Crossing-Rate", 0.01, 0.15),
    }

    idx = len(strict)

    msq = rpeak.get("MSQ (Multi-Detector)")
    if msq is not None and not (isinstance(msq, float) and (msq != msq)):
        idx += 1
        strict[idx] = _gt_rule("MSQ", 0.5)
        loose[idx] = _gt_rule("MSQ", 0.27)

    ectopic = rpeak.get("Ectopic Ratio (Malik)")
    if ectopic is not None and not (
        isinstance(ectopic, float) and (ectopic != ectopic)
    ):
        idx += 1
        strict[idx] = _lt_rule("Ectopic", 0.1)
        loose[idx] = _lt_rule("Ectopic", 0.3)

    hr_oor = hr.get("HR Out-of-Range (%, 40-200)")
    if hr_oor is not None and not (isinstance(hr_oor, float) and (hr_oor != hr_oor)):
        idx += 1
        strict[idx] = _lt_rule("HR-Range", 5.0)
        loose[idx] = _lt_rule("HR-Range", 20.0)

    return RuleSet(strict), RuleSet(loose)


def evaluate_quality(
    sqi_dict: Dict[str, float],
    strict_rs: RuleSet,
    loose_rs: RuleSet,
) -> str:
    """
    Classify signal quality as 'GOOD', 'FAIR', or 'POOR'.

    Parameters
    ----------
    sqi_dict  : dict of {rule_name: metric_value}
    strict_rs : RuleSet for GOOD classification
    loose_rs  : RuleSet for FAIR classification

    Returns
    -------
    'GOOD' | 'FAIR' | 'POOR'
    """
    df = pd.DataFrame([sqi_dict])
    try:
        if strict_rs.execute(df) == "accept":
            return "GOOD"
    except Exception:
        pass
    try:
        if loose_rs.execute(df) == "accept":
            return "FAIR"
    except Exception:
        pass
    return "POOR"


def make_sqi_dict(
    standard: Dict[str, float],
    rpeak: Dict[str, Any],
    hr: Dict[str, float],
) -> Dict[str, float]:
    """
    Build the flat SQI dict expected by evaluate_quality().

    Keys must match the rule names created in build_rulesets().
    """
    d: Dict[str, float] = {
        "Skewness": standard.get("Skewness", 0.0),
        "Kurtosis": standard.get("Kurtosis", 0.0),
        "SNR": standard.get("SNR", 0.0),
        "Perfusion": standard.get("Perfusion Index (%)", 0.0),
        "Zero-Crossing-Rate": standard.get("Zero Crossing Rate", 0.0),
        "Mean-Crossing-Rate": standard.get("Mean Crossing Rate", 0.0),
    }
    msq = rpeak.get("MSQ (Multi-Detector)")
    if msq is not None:
        d["MSQ"] = float(msq)
    ectopic = rpeak.get("Ectopic Ratio (Malik)")
    if ectopic is not None:
        d["Ectopic"] = float(ectopic)
    hr_oor = hr.get("HR Out-of-Range (%, 40-200)")
    if hr_oor is not None:
        d["HR-Range"] = float(hr_oor)
    return d

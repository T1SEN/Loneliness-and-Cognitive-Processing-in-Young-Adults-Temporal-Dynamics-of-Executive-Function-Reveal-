#!/usr/bin/env python3
"""
Reliability & Validity summary for online EF tasks
--------------------------------------------------
Outputs:
- results/analysis_outputs/rv_known_effects.csv
- results/analysis_outputs/rv_reliability.csv
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
RES = BASE / "results"
OUT = RES / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def _mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    ci = 1.96 * se if np.isfinite(se) else np.nan
    return m, m - ci, m + ci


def stroop_known_effects() -> dict:
    df = pd.read_csv(RES / "4c_stroop_trials.csv")
    # Choose RT column
    if "rt_ms" in df.columns and df["rt_ms"].notna().any():
        df["rt_ms2"] = pd.to_numeric(df["rt_ms"], errors="coerce")
    else:
        df["rt_ms2"] = pd.to_numeric(df.get("rt"), errors="coerce")
    df = df[(df.get("timeout", False) == False) & df["rt_ms2"].notna()]
    cond = df["cond"] if "cond" in df.columns else df.get("type")
    df = df.assign(cond=cond)
    # participant_id fallback
    df["participant_id"] = df["participant_id"].fillna(df.get("participantId"))
    # Per-participant congruent/incongruent means
    pivot = (
        df[df.get("correct", True) == True]
        .groupby(["participant_id", "cond"])['rt_ms2']
        .mean()
        .unstack()
    )
    if 'congruent' not in pivot.columns or 'incongruent' not in pivot.columns:
        return {"stroop_interference_ms": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    eff = pivot['incongruent'] - pivot['congruent']
    m, lo, hi = _mean_ci(eff.values)
    return {"stroop_interference_ms": m, "ci_low": lo, "ci_high": hi, "n": int(eff.notna().sum())}


def prp_known_effects() -> dict:
    df = pd.read_csv(RES / "4a_prp_trials.csv")
    # T2 RT column
    t2 = "t2_rt_ms" if "t2_rt_ms" in df.columns and df["t2_rt_ms"].notna().any() else "t2_rt"
    df[t2] = pd.to_numeric(df[t2], errors="coerce")
    df = df[(df.get("t2_timeout", False) == False) & df[t2].notna()]
    df["participant_id"] = df["participant_id"].fillna(df.get("participantId"))
    # SOA
    if "soa_nominal_ms" in df.columns and df["soa_nominal_ms"].notna().any():
        soa = pd.to_numeric(df["soa_nominal_ms"], errors="coerce")
    else:
        soa = pd.to_numeric(df.get("soa"), errors="coerce")
    df = df.assign(soa_ms=soa)
    short = df[df["soa_ms"] <= 150]
    long = df[df["soa_ms"] >= 1200]
    short_m = short.groupby("participant_id")[t2].mean()
    long_m = long.groupby("participant_id")[t2].mean()
    eff = (short_m - long_m).dropna()
    m, lo, hi = _mean_ci(eff.values)
    return {"prp_bottleneck_ms": m, "ci_low": lo, "ci_high": hi, "n": int(eff.notna().sum())}


def wcst_known_effects() -> dict:
    df = pd.read_csv(RES / "4b_wcst_trials.csv")
    df["participant_id"] = df["participant_id"].fillna(df.get("participantId"))
    df = df.sort_values(["participant_id", "trialIndex"]) if "trialIndex" in df.columns else df
    # Switch cost by rule change
    if "ruleAtThatTime" not in df.columns or "reactionTimeMs" not in df.columns:
        return {"wcst_switch_cost_ms": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    g = df.groupby("participant_id")
    rows = []
    for pid, grp in g:
        grp = grp.copy()
        grp["prev_rule"] = grp["ruleAtThatTime"].shift(1)
        grp["switch"] = (grp["ruleAtThatTime"] != grp["prev_rule"]) & grp["prev_rule"].notna()
        if grp["switch"].any() and (~grp["switch"]).any():
            sw = grp.loc[grp["switch"], "reactionTimeMs"].mean()
            ns = grp.loc[~grp["switch"], "reactionTimeMs"].mean()
            if np.isfinite(sw) and np.isfinite(ns):
                rows.append(sw - ns)
    if not rows:
        return {"wcst_switch_cost_ms": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    m, lo, hi = _mean_ci(np.array(rows))
    return {"wcst_switch_cost_ms": m, "ci_low": lo, "ci_high": hi, "n": len(rows)}


def stroop_reliability() -> dict:
    df = pd.read_csv(RES / "4c_stroop_trials.csv")
    df["participant_id"] = df["participant_id"].fillna(df.get("participantId"))
    if "rt_ms" in df.columns and df["rt_ms"].notna().any():
        df["rt_ms2"] = pd.to_numeric(df["rt_ms"], errors="coerce")
    else:
        df["rt_ms2"] = pd.to_numeric(df.get("rt"), errors="coerce")
    cond = df["cond"] if "cond" in df.columns else df.get("type")
    df = df.assign(cond=cond)
    df = df[(df.get("timeout", False) == False) & df["rt_ms2"].notna() & df["participant_id"].notna()]
    # split by odd/even trials
    halves = []
    for parity in [0, 1]:
        sub = df[df["trial"] % 2 == parity]
        pivot = (
            sub[sub.get("correct", True) == True]
            .groupby(["participant_id", "cond"])['rt_ms2']
            .mean()
            .unstack()
        )
        if 'congruent' not in pivot.columns or 'incongruent' not in pivot.columns:
            return {"stroop_split_half_r": np.nan, "sb": np.nan, "n": 0}
        eff = (pivot['incongruent'] - pivot['congruent']).rename(f"eff_{parity}")
        halves.append(eff)
    merged = pd.concat(halves, axis=1).dropna()
    if merged.empty:
        return {"stroop_split_half_r": np.nan, "sb": np.nan, "n": 0}
    r = np.corrcoef(merged.iloc[:, 0], merged.iloc[:, 1])[0, 1]
    sb = (2 * r) / (1 + r) if np.isfinite(r) else np.nan
    return {"stroop_split_half_r": r, "sb": sb, "n": int(len(merged))}


def prp_reliability() -> dict:
    df = pd.read_csv(RES / "4a_prp_trials.csv")
    df["participant_id"] = df["participant_id"].fillna(df.get("participantId"))
    t2 = "t2_rt_ms" if "t2_rt_ms" in df.columns and df["t2_rt_ms"].notna().any() else "t2_rt"
    df[t2] = pd.to_numeric(df[t2], errors="coerce")
    df = df[(df.get("t2_timeout", False) == False) & df[t2].notna()]
    # split by trial_index parity if exists, else row order
    if "trial_index" in df.columns and df["trial_index"].notna().any():
        df["half"] = df["trial_index"].astype(float) % 2
    else:
        df["half"] = np.arange(len(df)) % 2
    # compute bottleneck per half
    rows = []
    for h in [0, 1]:
        sub = df[df["half"] == h]
        if "soa_nominal_ms" in sub.columns and sub["soa_nominal_ms"].notna().any():
            soa = pd.to_numeric(sub["soa_nominal_ms"], errors="coerce")
        else:
            soa = pd.to_numeric(sub.get("soa"), errors="coerce")
        sub = sub.assign(soa_ms=soa)
        short = sub[sub["soa_ms"] <= 150].groupby("participant_id")[t2].mean()
        long = sub[sub["soa_ms"] >= 1200].groupby("participant_id")[t2].mean()
        eff = (short - long).rename(f"eff_{h}")
        rows.append(eff)
    merged = pd.concat(rows, axis=1).dropna()
    if merged.empty:
        return {"prp_split_half_r": np.nan, "sb": np.nan, "n": 0}
    r = np.corrcoef(merged.iloc[:, 0], merged.iloc[:, 1])[0, 1]
    sb = (2 * r) / (1 + r) if np.isfinite(r) else np.nan
    return {"prp_split_half_r": r, "sb": sb, "n": int(len(merged))}


def wcst_reliability() -> dict:
    df = pd.read_csv(RES / "4b_wcst_trials.csv")
    df["participant_id"] = df["participant_id"].fillna(df.get("participantId"))
    if "blockIndex" not in df.columns:
        return {"wcst_deck_r": np.nan, "n": 0}
    def metric(sub):
        # total error rate per block
        if "correct" not in sub.columns:
            return np.nan
        return 1 - sub["correct"].mean()
    g = df.groupby(["participant_id", "blockIndex"]).apply(metric).unstack()
    if g.shape[1] < 2:
        return {"wcst_deck_r": np.nan, "n": 0}
    g = g.dropna()
    if g.empty:
        return {"wcst_deck_r": np.nan, "n": 0}
    r = np.corrcoef(g.iloc[:, 0], g.iloc[:, 1])[0, 1]
    return {"wcst_deck_r": r, "n": int(len(g))}


def main():
    known = {**stroop_known_effects(), **prp_known_effects(), **wcst_known_effects()}
    rel = {**stroop_reliability(), **prp_reliability(), **wcst_reliability()}
    pd.DataFrame([known]).to_csv(OUT / "rv_known_effects.csv", index=False)
    pd.DataFrame([rel]).to_csv(OUT / "rv_reliability.csv", index=False)
    print("Known effects:")
    print(pd.DataFrame([known]).to_string(index=False))
    print("\nReliability:")
    print(pd.DataFrame([rel]).to_string(index=False))


if __name__ == "__main__":
    main()

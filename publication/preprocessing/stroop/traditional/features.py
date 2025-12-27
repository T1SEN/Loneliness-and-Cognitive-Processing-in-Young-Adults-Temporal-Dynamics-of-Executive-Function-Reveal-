"""Stroop traditional feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ...constants import STROOP_RT_MIN
from .._shared import _ez_ddm_params, prepare_stroop_trials


def derive_stroop_traditional_features(
    rt_min: float = STROOP_RT_MIN,
    rt_max: float | None = None,
    data_dir: None | str | Path = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_stroop_trials(rt_min=rt_min, rt_max=rt_max, data_dir=data_dir)

    stroop = prepared["stroop"]
    stroop_acc = prepared["stroop_acc"]
    cond_col = prepared["cond_col"]

    if not isinstance(stroop, pd.DataFrame) or stroop.empty:
        return pd.DataFrame()

    def _ez_v_sd_by_quartile(df: pd.DataFrame) -> float:
        if df.empty:
            return np.nan
        if "trial_order" in df.columns:
            df = df.sort_values("trial_order")
        df = df.reset_index(drop=True)
        if len(df) < 20:
            return np.nan
        try:
            df["quartile"] = pd.qcut(
                range(len(df)),
                q=4,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            return np.nan
        v_vals = []
        for q in sorted(df["quartile"].dropna().unique()):
            q_df = df[df["quartile"] == q]
            if q_df.empty:
                continue
            acc = q_df["correct"].mean() if "correct" in q_df.columns else np.nan
            if "correct" in q_df.columns:
                rt_corr = q_df[q_df["correct"] == True]["rt_ms"]
            else:
                rt_corr = q_df["rt_ms"]
            params = _ez_ddm_params(rt_corr, acc, int(len(q_df)))
            v = params.get("v", np.nan)
            if pd.notna(v):
                v_vals.append(float(v))
        if len(v_vals) >= 2:
            return float(np.std(v_vals, ddof=1)) if len(v_vals) > 1 else 0.0
        return np.nan

    acc_groups = {}
    if isinstance(stroop_acc, pd.DataFrame) and not stroop_acc.empty:
        acc_groups = {pid: grp for pid, grp in stroop_acc.groupby("participant_id")}

    records: List[Dict] = []

    for pid, group in stroop.groupby("participant_id"):
        grp = group.copy()
        acc_grp = acc_groups.get(pid, pd.DataFrame())

        rt_correct = grp[grp["correct"] == True] if "correct" in grp.columns else grp

        rt_interference = np.nan
        rt_facilitation = np.nan
        rt_incong_neutral = np.nan
        rt_neutral_cong = np.nan
        rt_interference_correct = np.nan
        rt_facilitation_correct = np.nan
        rt_incong_neutral_correct = np.nan
        rt_neutral_cong_correct = np.nan
        acc_interference = np.nan
        acc_facilitation = np.nan

        if cond_col:
            for df, suffix in ((grp, ""), (rt_correct, "_correct")):
                rt_cong = df.loc[df["condition_norm"] == "congruent", "rt_ms"].mean()
                rt_incong = df.loc[df["condition_norm"] == "incongruent", "rt_ms"].mean()
                rt_neutral = df.loc[df["condition_norm"] == "neutral", "rt_ms"].mean()

                if pd.notna(rt_incong) and pd.notna(rt_cong):
                    if suffix:
                        rt_interference_correct = rt_incong - rt_cong
                    else:
                        rt_interference = rt_incong - rt_cong
                if pd.notna(rt_neutral) and pd.notna(rt_cong):
                    if suffix:
                        rt_facilitation_correct = rt_neutral - rt_cong
                        rt_neutral_cong_correct = rt_neutral - rt_cong
                    else:
                        rt_facilitation = rt_neutral - rt_cong
                        rt_neutral_cong = rt_neutral - rt_cong
                if pd.notna(rt_incong) and pd.notna(rt_neutral):
                    if suffix:
                        rt_incong_neutral_correct = rt_incong - rt_neutral
                    else:
                        rt_incong_neutral = rt_incong - rt_neutral

            if not acc_grp.empty and "correct" in acc_grp.columns:
                acc_cong = acc_grp.loc[acc_grp["condition_norm"] == "congruent", "correct"].mean()
                acc_incong = acc_grp.loc[acc_grp["condition_norm"] == "incongruent", "correct"].mean()
                acc_neutral = acc_grp.loc[acc_grp["condition_norm"] == "neutral", "correct"].mean()
                if pd.notna(acc_incong) and pd.notna(acc_cong):
                    acc_interference = acc_incong - acc_cong
                if pd.notna(acc_neutral) and pd.notna(acc_cong):
                    acc_facilitation = acc_neutral - acc_cong

        ez_params = {}
        if cond_col:
            for cond in ("congruent", "incongruent", "neutral"):
                cond_mask = grp["condition_norm"] == cond
                n_trials = int(cond_mask.sum())
                if "correct" in grp.columns:
                    acc = grp.loc[cond_mask, "correct"].mean()
                    rt_corr = grp.loc[cond_mask & (grp["correct"] == True), "rt_ms"]
                else:
                    acc = np.nan
                    rt_corr = pd.Series(dtype=float)
                params = _ez_ddm_params(rt_corr, acc, n_trials)
                ez_params[cond] = params

        records.append({
            "participant_id": pid,
            "stroop_rt_interference": rt_interference,
            "stroop_rt_facilitation": rt_facilitation,
            "stroop_rt_incong_minus_neutral": rt_incong_neutral,
            "stroop_rt_neutral_minus_cong": rt_neutral_cong,
            "stroop_rt_interference_correct": rt_interference_correct,
            "stroop_rt_facilitation_correct": rt_facilitation_correct,
            "stroop_rt_incong_minus_neutral_correct": rt_incong_neutral_correct,
            "stroop_rt_neutral_minus_cong_correct": rt_neutral_cong_correct,
            "stroop_acc_interference": acc_interference,
            "stroop_acc_facilitation": acc_facilitation,
            "stroop_trials": len(grp),
        })

        if ez_params:
            record = records[-1]
            for cond in ("congruent", "incongruent", "neutral"):
                params = ez_params.get(cond, {})
                record[f"stroop_ez_{cond}_v"] = params.get("v", np.nan)
                record[f"stroop_ez_{cond}_a"] = params.get("a", np.nan)
                record[f"stroop_ez_{cond}_t0"] = params.get("t0", np.nan)
                record[f"stroop_ez_{cond}_sv"] = 0.0
                record[f"stroop_ez_{cond}_sz"] = 0.0
                record[f"stroop_ez_{cond}_st0"] = 0.0

            if cond_col:
                for cond in ("congruent", "incongruent", "neutral"):
                    cond_df = grp[grp["condition_norm"] == cond]
                    record[f"stroop_ez_{cond}_v_sd"] = _ez_v_sd_by_quartile(cond_df)

                v_sd_incong = record.get("stroop_ez_incongruent_v_sd")
                v_sd_cong = record.get("stroop_ez_congruent_v_sd")
                if pd.notna(v_sd_incong) and pd.notna(v_sd_cong):
                    record["stroop_ez_v_sd_interference"] = v_sd_incong - v_sd_cong
                else:
                    record["stroop_ez_v_sd_interference"] = np.nan

            if (
                pd.notna(record.get("stroop_ez_incongruent_v"))
                and pd.notna(record.get("stroop_ez_congruent_v"))
            ):
                record["stroop_ez_v_interference"] = (
                    record["stroop_ez_incongruent_v"] - record["stroop_ez_congruent_v"]
                )
            else:
                record["stroop_ez_v_interference"] = np.nan

            if (
                pd.notna(record.get("stroop_ez_incongruent_a"))
                and pd.notna(record.get("stroop_ez_congruent_a"))
            ):
                record["stroop_ez_a_interference"] = (
                    record["stroop_ez_incongruent_a"] - record["stroop_ez_congruent_a"]
                )
            else:
                record["stroop_ez_a_interference"] = np.nan

            if (
                pd.notna(record.get("stroop_ez_incongruent_t0"))
                and pd.notna(record.get("stroop_ez_congruent_t0"))
            ):
                record["stroop_ez_t0_interference"] = (
                    record["stroop_ez_incongruent_t0"] - record["stroop_ez_congruent_t0"]
                )
            else:
                record["stroop_ez_t0_interference"] = np.nan

    return pd.DataFrame(records)

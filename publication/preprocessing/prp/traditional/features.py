"""PRP traditional feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from ...constants import DEFAULT_SOA_LONG, DEFAULT_SOA_SHORT, PRP_RT_MIN
from .._shared import prepare_prp_trials


def derive_prp_traditional_features(
    rt_min: float = PRP_RT_MIN,
    rt_max: float | None = None,
    data_dir: Path | None = None,
    prepared: Dict[str, object] | None = None,
) -> pd.DataFrame:
    if prepared is None:
        prepared = prepare_prp_trials(rt_min=rt_min, rt_max=rt_max, data_dir=data_dir)

    prp = prepared["prp"]
    prp_acc = prepared.get("prp_acc", prp)
    prp_raw = prepared["prp_raw"]
    soa_levels = prepared["soa_levels"]

    if not isinstance(prp, pd.DataFrame) or prp.empty:
        return pd.DataFrame()

    raw_groups = {}
    if isinstance(prp_raw, pd.DataFrame) and not prp_raw.empty:
        raw_groups = {pid: grp for pid, grp in prp_raw.groupby("participant_id")}

    acc_groups = {pid: grp for pid, grp in prp_acc.groupby("participant_id")} if not prp_acc.empty else {}
    records: List[Dict] = []

    for pid, group in prp.groupby("participant_id"):
        group_acc = acc_groups.get(pid, pd.DataFrame())
        raw_grp = raw_groups.get(pid, pd.DataFrame())

        order_violation_rate = np.nan
        t1_only_rate = np.nan
        t2_only_rate = np.nan
        t2_pending_rate = np.nan
        iri_mean = np.nan
        iri_median = np.nan
        iri_p10 = np.nan
        iri_p25 = np.nan
        iri_grouping_rate_100 = np.nan

        if not raw_grp.empty:
            if "order_norm" in raw_grp.columns:
                order_counts = raw_grp["order_norm"].value_counts(dropna=True)
                total_orders = int(order_counts.sum())
                if total_orders > 0:
                    order_violation_rate = order_counts.get("t2_t1", 0) / total_orders
                    t1_only_rate = order_counts.get("t1only", 0) / total_orders
                    t2_only_rate = order_counts.get("t2only", 0) / total_orders

            if "t2_pressed_while_t1_pending" in raw_grp.columns:
                t2_pending_rate = float(pd.to_numeric(raw_grp["t2_pressed_while_t1_pending"], errors="coerce").mean())

            iri_df = raw_grp.copy()
            for timeout_col in ("t1_timeout", "t2_timeout"):
                if timeout_col in iri_df.columns:
                    iri_df = iri_df[iri_df[timeout_col] == False]
            if "order_norm" in iri_df.columns:
                iri_df = iri_df[iri_df["order_norm"] == "t1_t2"]
            if "t2_pressed_while_t1_pending" in iri_df.columns:
                iri_df = iri_df[iri_df["t2_pressed_while_t1_pending"] == False]
            if "iri_ms" in iri_df.columns:
                iri_vals = pd.to_numeric(iri_df["iri_ms"], errors="coerce").dropna()
                if len(iri_vals) > 0:
                    iri_vals = iri_vals[iri_vals >= 0]
                    iri_mean = float(iri_vals.mean())
                    iri_median = float(iri_vals.median())
                    iri_p10 = float(iri_vals.quantile(0.1))
                    iri_p25 = float(iri_vals.quantile(0.25))
                    iri_grouping_rate_100 = float((iri_vals < 100).mean())

        t1_cost = np.nan
        if "t1_rt_ms" in group.columns:
            t1_short = group[group["soa_ms"] <= DEFAULT_SOA_SHORT]["t1_rt_ms"].mean()
            t1_long = group[group["soa_ms"] >= DEFAULT_SOA_LONG]["t1_rt_ms"].mean()
            if pd.notna(t1_short) and pd.notna(t1_long):
                t1_cost = t1_short - t1_long

        rt1_rt2_coupling = np.nan
        if "t1_rt_ms" in group.columns:
            pair = group[["t1_rt_ms", "t2_rt_ms"]].dropna()
            if len(pair) >= 3:
                rt1_rt2_coupling = float(np.corrcoef(pair["t1_rt_ms"], pair["t2_rt_ms"])[0, 1])

        t2_acc_short = np.nan
        t2_acc_long = np.nan
        t2_acc_cost = np.nan
        if not group_acc.empty and "t2_correct" in group_acc.columns:
            t2_acc_short = group_acc[group_acc["soa_ms"] <= DEFAULT_SOA_SHORT]["t2_correct"].mean()
            t2_acc_long = group_acc[group_acc["soa_ms"] >= DEFAULT_SOA_LONG]["t2_correct"].mean()
            if pd.notna(t2_acc_short) and pd.notna(t2_acc_long):
                t2_acc_cost = t2_acc_long - t2_acc_short

        records.append({
            "participant_id": pid,
            "prp_t1_cost": t1_cost,
            "prp_rt1_rt2_coupling": rt1_rt2_coupling,
            "prp_order_violation_rate": order_violation_rate,
            "prp_t1_only_rate": t1_only_rate,
            "prp_t2_only_rate": t2_only_rate,
            "prp_t2_while_t1_pending_rate": t2_pending_rate,
            "prp_iri_mean": iri_mean,
            "prp_iri_median": iri_median,
            "prp_iri_p10": iri_p10,
            "prp_iri_p25": iri_p25,
            "prp_iri_grouping_rate_100": iri_grouping_rate_100,
            "prp_t2_acc_short": t2_acc_short,
            "prp_t2_acc_long": t2_acc_long,
            "prp_t2_acc_cost": t2_acc_cost,
            "prp_t2_trials": len(group),
        })

        if soa_levels:
            record = records[-1]
            for soa_val in soa_levels:
                label = int(round(float(soa_val)))
                subset_rt = group[group["soa_ms"] == soa_val]
                record[f"prp_t2_rt_mean_soa_{label}"] = subset_rt["t2_rt_ms"].mean()
                record[f"prp_t2_rt_median_soa_{label}"] = subset_rt["t2_rt_ms"].median()
                if "t1_rt_ms" in subset_rt.columns:
                    record[f"prp_t1_rt_mean_soa_{label}"] = subset_rt["t1_rt_ms"].mean()
                    record[f"prp_t1_rt_median_soa_{label}"] = subset_rt["t1_rt_ms"].median()
                if not group_acc.empty:
                    subset_acc = group_acc[group_acc["soa_ms"] == soa_val]
                    if "t1_correct" in subset_acc.columns:
                        record[f"prp_t1_acc_soa_{label}"] = subset_acc["t1_correct"].mean()
                    if "t2_correct" in subset_acc.columns:
                        record[f"prp_t2_acc_soa_{label}"] = subset_acc["t2_correct"].mean()
                    if "t1_correct" in subset_acc.columns and "t2_correct" in subset_acc.columns:
                        record[f"prp_both_correct_soa_{label}"] = (
                            subset_acc["t1_correct"] & subset_acc["t2_correct"]
                        ).mean()
                if not raw_grp.empty:
                    subset_raw = raw_grp[raw_grp["soa_ms"] == soa_val]
                    timeout_mask = None
                    if "t1_timeout" in subset_raw.columns:
                        timeout_mask = subset_raw["t1_timeout"].copy()
                    if "t2_timeout" in subset_raw.columns:
                        timeout_mask = (
                            timeout_mask | subset_raw["t2_timeout"]
                            if timeout_mask is not None
                            else subset_raw["t2_timeout"]
                        )
                    if timeout_mask is not None and len(subset_raw) > 0:
                        record[f"prp_timeout_rate_soa_{label}"] = float(timeout_mask.mean())
                    if "order_norm" in subset_raw.columns:
                        order_vals = subset_raw["order_norm"].dropna()
                        record[f"prp_order_error_soa_{label}"] = (
                            float((order_vals == "t2_t1").mean())
                            if len(order_vals) > 0
                            else np.nan
                        )
                if "t1_rt_ms" in subset_rt.columns:
                    pair = subset_rt[["t1_rt_ms", "t2_rt_ms"]].dropna()
                    if len(pair) >= 3:
                        record[f"prp_rt1_rt2_coupling_soa_{label}"] = float(
                            np.corrcoef(pair["t1_rt_ms"], pair["t2_rt_ms"])[0, 1]
                        )
                    else:
                        record[f"prp_rt1_rt2_coupling_soa_{label}"] = np.nan

    return pd.DataFrame(records)

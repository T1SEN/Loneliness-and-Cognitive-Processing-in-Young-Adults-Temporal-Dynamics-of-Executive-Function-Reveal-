"""PRP bottleneck mechanism feature derivation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..constants import PRP_IRI_MIN, PRP_RT_MIN, PRP_RT_MAX, get_results_dir
from .loaders import load_prp_trials

MECHANISM_FILENAME = "5_prp_bottleneck_mechanism_features.csv"


def _cbt_model(soa: np.ndarray, base: float, bottleneck: float) -> np.ndarray:
    return base + np.maximum(0.0, bottleneck - soa)


def _fit_cbt_model(soa_values: np.ndarray, rt_values: np.ndarray) -> Dict[str, float]:
    if len(soa_values) < 3:
        return {"base": np.nan, "bottleneck": np.nan, "r2": np.nan, "rmse": np.nan}

    order = np.argsort(soa_values)
    soa_sorted = soa_values[order]
    rt_sorted = rt_values[order]

    base_init = float(np.mean(rt_sorted[-min(2, len(rt_sorted)) :]))
    min_idx = int(np.argmin(soa_sorted))
    min_soa = float(soa_sorted[min_idx])
    min_rt = float(rt_sorted[min_idx])
    bottleneck_init = max(50.0, min(1500.0, min_soa + max(0.0, min_rt - base_init)))

    try:
        params, _ = curve_fit(
            _cbt_model,
            soa_sorted,
            rt_sorted,
            p0=[base_init, bottleneck_init],
            bounds=([PRP_RT_MIN, 50.0], [PRP_RT_MAX, 1500.0]),
            maxfev=10000,
        )
        base, bottleneck = params
        preds = _cbt_model(soa_sorted, base, bottleneck)
        ss_res = float(np.sum((rt_sorted - preds) ** 2))
        ss_tot = float(np.sum((rt_sorted - np.mean(rt_sorted)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(ss_res / len(rt_sorted)))
        return {"base": float(base), "bottleneck": float(bottleneck), "r2": r2, "rmse": rmse}
    except Exception:
        return {"base": np.nan, "bottleneck": np.nan, "r2": np.nan, "rmse": np.nan}


def compute_prp_bottleneck_features(
    data_dir: Path | None = None,
    min_trials: int = 30,
    min_soa_levels: int = 3,
    short_soa_max: int = 300,
) -> pd.DataFrame:
    trials, _ = load_prp_trials(
        data_dir=data_dir,
        rt_min=PRP_RT_MIN,
        rt_max=PRP_RT_MAX,
        require_t1_correct=True,
        require_t2_correct_for_rt=True,
        enforce_short_long_only=False,
        drop_timeouts=True,
        require_valid_order=True,
        iri_min=PRP_IRI_MIN,
        exclude_t2_pressed_while_pending=True,
    )

    trials["t2_rt"] = pd.to_numeric(trials["t2_rt"], errors="coerce")
    trials["soa"] = pd.to_numeric(trials["soa"], errors="coerce")
    trials = trials[trials["t2_rt"].notna() & trials["soa"].notna()]

    results = []
    for pid, group in trials.groupby("participant_id"):
        n_trials = int(len(group))
        agg = (
            group.groupby("soa")
            .agg(rt_mean=("t2_rt", "mean"), n=("t2_rt", "count"))
            .reset_index()
        )
        n_soa_levels = int(agg["soa"].nunique())
        record = {
            "participant_id": pid,
            "prp_cb_base": np.nan,
            "prp_cb_bottleneck": np.nan,
            "prp_cb_r_squared": np.nan,
            "prp_cb_rmse": np.nan,
            "prp_cb_slope": np.nan,
            "prp_cb_n_trials": n_trials,
            "prp_cb_n_soa_levels": n_soa_levels,
        }

        if n_trials < min_trials or n_soa_levels < min_soa_levels:
            results.append(record)
            continue

        fit = _fit_cbt_model(agg["soa"].to_numpy(), agg["rt_mean"].to_numpy())
        record["prp_cb_base"] = fit["base"]
        record["prp_cb_bottleneck"] = fit["bottleneck"]
        record["prp_cb_r_squared"] = fit["r2"]
        record["prp_cb_rmse"] = fit["rmse"]

        short = agg[agg["soa"] <= short_soa_max]
        if len(short) >= 2:
            slope, _ = np.polyfit(short["soa"], short["rt_mean"], 1)
            record["prp_cb_slope"] = float(slope)

        results.append(record)

    return pd.DataFrame(results)


def load_or_compute_prp_bottleneck_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("prp")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_prp_bottleneck_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] PRP bottleneck features saved: {output_path}")
    return features

"""PRP bottleneck mechanism feature derivation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..constants import DEFAULT_SOA_LONG, PRP_RT_MIN, PRP_RT_MAX, get_results_dir
from .loaders import load_prp_trials

MECHANISM_FILENAME = "5_prp_bottleneck_mechanism_features.csv"


def _pick_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_response_order(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    token = "".join(ch for ch in cleaned if ch.isalnum())
    if token.startswith("t1t2"):
        return "t1_t2"
    if token.startswith("t2t1"):
        return "t2_t1"
    if token in {"t1only", "t2only", "none"}:
        return token
    return None


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    })
    return mapped.fillna(False).astype(bool)


@dataclass(frozen=True)
class PrpCols:
    pid: str
    soa: str
    rt2: str
    t2_correct: Optional[str]
    t2_timeout: Optional[str]
    response_order: Optional[str]


def _resolve_prp_columns(df: pd.DataFrame) -> PrpCols:
    pid = _pick_col(df, ("participant_id", "participantId", "pid", "subject_id", "subj_id"))
    soa = _pick_col(df, ("soa_nominal_ms", "soa", "soa_ms", "SOA", "soa_nominal"))
    rt2 = _pick_col(df, ("t2_rt_ms", "t2_rt", "rt2", "t2rt", "T2_RT"))

    if pid is None or soa is None or rt2 is None:
        raise ValueError(
            "Required columns not found. Need participant_id, SOA, and T2 RT.\n"
            f"Available columns: {list(df.columns)}"
        )

    t2_correct = _pick_col(df, ("t2_correct", "T2_correct", "t2_is_correct"))
    t2_timeout = _pick_col(df, ("t2_timeout", "T2_timeout", "t2_is_timeout"))
    response_order = _pick_col(df, ("response_order", "resp_order", "order", "order_norm"))

    return PrpCols(
        pid=pid,
        soa=soa,
        rt2=rt2,
        t2_correct=t2_correct,
        t2_timeout=t2_timeout,
        response_order=response_order,
    )


def _cbt_model(soa: np.ndarray, base: float, bottleneck: float) -> np.ndarray:
    return base + np.maximum(0.0, bottleneck - soa)


def _cs_model(soa: np.ndarray, base: float, amplitude: float, tau: float) -> np.ndarray:
    return base + amplitude * np.exp(-soa / tau)


def _fit_cbt_model(
    soa_values: np.ndarray,
    rt_values: np.ndarray,
    rt_se: Optional[np.ndarray] = None,
    rt_min: float = PRP_RT_MIN,
    rt_max: float = PRP_RT_MAX,
    bottleneck_min: float = 50.0,
    bottleneck_max: float = 1500.0,
    maxfev: int = 20000,
) -> Dict[str, float]:
    if len(soa_values) < 3:
        return {"base": np.nan, "bottleneck": np.nan, "r2": np.nan, "rmse": np.nan}

    order = np.argsort(soa_values)
    soa_sorted = soa_values[order].astype(float)
    rt_sorted = rt_values[order].astype(float)

    sigma = None
    if rt_se is not None:
        se_sorted = rt_se[order].astype(float)
        if np.all(np.isfinite(se_sorted)) and np.all(se_sorted > 0):
            sigma = se_sorted

    base_init = float(np.mean(rt_sorted[-min(2, len(rt_sorted)) :]))
    min_idx = int(np.argmin(soa_sorted))
    min_soa = float(soa_sorted[min_idx])
    min_rt = float(rt_sorted[min_idx])
    bottleneck_init = float(
        np.clip(min_soa + max(0.0, min_rt - base_init), bottleneck_min, bottleneck_max)
    )

    try:
        params, _ = curve_fit(
            _cbt_model,
            soa_sorted,
            rt_sorted,
            p0=[base_init, bottleneck_init],
            bounds=([rt_min, bottleneck_min], [rt_max, bottleneck_max]),
            sigma=sigma,
            absolute_sigma=True if sigma is not None else False,
            maxfev=maxfev,
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


def _fit_cs_model(
    soa_values: np.ndarray,
    rt_values: np.ndarray,
    rt_min: float = PRP_RT_MIN,
    rt_max: float = PRP_RT_MAX,
    maxfev: int = 20000,
) -> Dict[str, float]:
    if len(soa_values) < 3:
        return {"base": np.nan, "amplitude": np.nan, "tau": np.nan, "r2": np.nan, "rmse": np.nan}

    order = np.argsort(soa_values)
    soa_sorted = soa_values[order].astype(float)
    rt_sorted = rt_values[order].astype(float)

    base_init = float(np.mean(rt_sorted[-min(2, len(rt_sorted)) :]))
    amp_init = float(max(0.0, np.max(rt_sorted) - base_init))
    tau_init = float(np.clip(np.median(soa_sorted), 50.0, 2000.0))

    try:
        params, _ = curve_fit(
            _cs_model,
            soa_sorted,
            rt_sorted,
            p0=[base_init, amp_init, tau_init],
            bounds=([rt_min, 0.0, 50.0], [rt_max, rt_max, 5000.0]),
            maxfev=maxfev,
        )
        base, amp, tau = params
        preds = _cs_model(soa_sorted, base, amp, tau)
        ss_res = float(np.sum((rt_sorted - preds) ** 2))
        ss_tot = float(np.sum((rt_sorted - np.mean(rt_sorted)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = float(np.sqrt(ss_res / len(rt_sorted)))
        return {
            "base": float(base),
            "amplitude": float(amp),
            "tau": float(tau),
            "r2": r2,
            "rmse": rmse,
        }
    except Exception:
        return {"base": np.nan, "amplitude": np.nan, "tau": np.nan, "r2": np.nan, "rmse": np.nan}


def _compute_aic_bic(rss: float, n: int, k: int) -> Tuple[float, float]:
    if n <= 0 or not np.isfinite(rss) or rss <= 0:
        return np.nan, np.nan
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return float(aic), float(bic)


def compute_prp_bottleneck_features(
    data_dir: Path | None = None,
    min_trials: int = 30,
    min_soa_levels: int = 3,
    short_soa_max: int = 300,
    min_trials_per_soa: int = 3,
    use_measured_soa: bool = False,
    include_incorrect_trials_for_curve: bool = True,
    require_t1_to_t2_order: bool = False,
    apply_trial_filters: bool = True,
    apply_rt_bounds: bool = False,
    maxfev: int = 20000,
) -> pd.DataFrame:
    trials, _ = load_prp_trials(
        data_dir=data_dir,
        apply_trial_filters=apply_trial_filters,
    )
    cols = _resolve_prp_columns(trials)

    df = trials.copy()

    if use_measured_soa:
        measured = _pick_col(df, ("soa_measured_ms", "soa_measured"))
        if measured is not None:
            df[measured] = pd.to_numeric(df[measured], errors="coerce")
            df[cols.soa] = (df[measured] / 10.0).round() * 10.0

    df[cols.rt2] = pd.to_numeric(df[cols.rt2], errors="coerce")
    df[cols.soa] = pd.to_numeric(df[cols.soa], errors="coerce")
    df = df[df[cols.rt2].notna() & df[cols.soa].notna()].copy()

    if apply_rt_bounds:
        df = df[(df[cols.rt2] >= PRP_RT_MIN) & (df[cols.rt2] <= PRP_RT_MAX)].copy()

    if cols.t2_timeout is not None:
        df[cols.t2_timeout] = _coerce_bool_series(df[cols.t2_timeout])
        df = df[df[cols.t2_timeout] == False].copy()  # noqa: E712

    if require_t1_to_t2_order and cols.response_order is not None:
        if cols.response_order == "order_norm":
            order_norm = df["order_norm"]
        else:
            order_norm = df[cols.response_order].apply(_normalize_response_order)
            df["order_norm"] = order_norm
        df = df[order_norm == "t1_t2"].copy()

    if (not include_incorrect_trials_for_curve) and cols.t2_correct is not None:
        df[cols.t2_correct] = _coerce_bool_series(df[cols.t2_correct])
        df_curve = df[df[cols.t2_correct] == True].copy()  # noqa: E712
    else:
        df_curve = df.copy()

    df_short = df_curve[df_curve[cols.soa] <= float(short_soa_max)].copy()

    results = []
    for pid, g_all in df.groupby(cols.pid):
        g_curve = df_curve[df_curve[cols.pid] == pid]
        g_short = df_short[df_short[cols.pid] == pid]

        record = {
            "participant_id": pid,
            "prp_cb_base": np.nan,
            "prp_cb_bottleneck": np.nan,
            "prp_cb_r_squared": np.nan,
            "prp_cb_rmse": np.nan,
            "prp_cb_slope": np.nan,
            "prp_cb_n_trials": int(len(g_curve)),
            "prp_cb_n_soa_levels": int(g_curve[cols.soa].nunique()) if len(g_curve) else 0,
        }

        if len(g_curve) < min_trials or g_curve[cols.soa].nunique() < min_soa_levels:
            results.append(record)
            continue

        agg = (
            g_curve.groupby(cols.soa)[cols.rt2]
            .agg(rt_mean="mean", rt_sd="std", n="count")
            .reset_index()
            .rename(columns={cols.soa: "soa"})
        )
        agg = agg[agg["n"] >= int(min_trials_per_soa)].copy()

        if len(agg) < min_soa_levels:
            results.append(record)
            continue

        agg["rt_se"] = agg["rt_sd"] / np.sqrt(agg["n"].clip(lower=1))

        fit = _fit_cbt_model(
            agg["soa"].to_numpy(),
            agg["rt_mean"].to_numpy(),
            rt_se=agg["rt_se"].to_numpy(),
            maxfev=maxfev,
        )
        record["prp_cb_base"] = fit["base"]
        record["prp_cb_bottleneck"] = fit["bottleneck"]
        record["prp_cb_r_squared"] = fit["r2"]
        record["prp_cb_rmse"] = fit["rmse"]

        soa_vals = agg["soa"].to_numpy()
        rt_means = agg["rt_mean"].to_numpy()
        if len(soa_vals) >= 2:
            record["prp_bottleneck_auc"] = float(np.trapz(rt_means, soa_vals))
        else:
            record["prp_bottleneck_auc"] = np.nan

        short_range = agg[agg["soa"] <= 300]
        if len(short_range) >= 2:
            slope, _ = np.polyfit(short_range["soa"], short_range["rt_mean"], 1)
            record["prp_bottleneck_slope_short"] = float(slope)
        else:
            record["prp_bottleneck_slope_short"] = np.nan

        long_range = agg[(agg["soa"] >= 300) & (agg["soa"] <= DEFAULT_SOA_LONG)]
        if len(long_range) >= 2:
            slope, _ = np.polyfit(long_range["soa"], long_range["rt_mean"], 1)
            record["prp_bottleneck_slope_long"] = float(slope)
        else:
            record["prp_bottleneck_slope_long"] = np.nan

        long_soa = agg[agg["soa"] >= DEFAULT_SOA_LONG]
        record["prp_rt2_asymptote_long_soa"] = float(long_soa["rt_mean"].mean()) if len(long_soa) else np.nan

        if len(agg) >= 3:
            coef = np.polyfit(agg["soa"], agg["rt_mean"], 2)
            record["prp_bottleneck_curvature"] = float(coef[0])
        else:
            record["prp_bottleneck_curvature"] = np.nan

        short = agg[agg["soa"] <= short_soa_max]
        if len(short) >= 2:
            slope, _ = np.polyfit(short["soa"], short["rt_mean"], 1)
            record["prp_cb_slope"] = float(slope)

        if pd.notna(record["prp_cb_base"]) and pd.notna(record["prp_cb_bottleneck"]):
            preds = _cbt_model(soa_vals, record["prp_cb_base"], record["prp_cb_bottleneck"])
            rss = float(np.sum((rt_means - preds) ** 2))
            aic, bic = _compute_aic_bic(rss, len(soa_vals), 2)
            record["prp_cb_aic"] = aic
            record["prp_cb_bic"] = bic
        else:
            record["prp_cb_aic"] = np.nan
            record["prp_cb_bic"] = np.nan

        cs_fit = _fit_cs_model(soa_vals, rt_means, maxfev=maxfev)
        record["prp_cs_base"] = cs_fit["base"]
        record["prp_cs_amplitude"] = cs_fit["amplitude"]
        record["prp_cs_tau"] = cs_fit["tau"]
        record["prp_cs_r_squared"] = cs_fit["r2"]
        record["prp_cs_rmse"] = cs_fit["rmse"]

        if pd.notna(record["prp_cs_base"]) and pd.notna(record["prp_cs_amplitude"]) and pd.notna(record["prp_cs_tau"]):
            preds = _cs_model(soa_vals, record["prp_cs_base"], record["prp_cs_amplitude"], record["prp_cs_tau"])
            rss = float(np.sum((rt_means - preds) ** 2))
            aic, bic = _compute_aic_bic(rss, len(soa_vals), 3)
            record["prp_cs_aic"] = aic
            record["prp_cs_bic"] = bic
        else:
            record["prp_cs_aic"] = np.nan
            record["prp_cs_bic"] = np.nan

        if (
            pd.notna(record["prp_cb_base"])
            and pd.notna(record["prp_cb_bottleneck"])
            and pd.notna(record["prp_cs_base"])
            and pd.notna(record["prp_cs_amplitude"])
            and pd.notna(record["prp_cs_tau"])
        ):
            pred_cb = _cbt_model(soa_vals, record["prp_cb_base"], record["prp_cb_bottleneck"])
            pred_cs = _cs_model(soa_vals, record["prp_cs_base"], record["prp_cs_amplitude"], record["prp_cs_tau"])
            diff = pred_cb - pred_cs
            denom = float(np.sum(diff ** 2))
            if denom > 0:
                weight = float(np.sum((rt_means - pred_cs) * diff) / denom)
                weight = float(np.clip(weight, 0.0, 1.0))
                pred_mix = weight * pred_cb + (1.0 - weight) * pred_cs
                rss = float(np.sum((rt_means - pred_mix) ** 2))
                ss_tot = float(np.sum((rt_means - np.mean(rt_means)) ** 2))
                r2 = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan
                rmse = float(np.sqrt(rss / len(rt_means)))
                aic, bic = _compute_aic_bic(rss, len(soa_vals), 6)
            else:
                weight = np.nan
                r2 = np.nan
                rmse = np.nan
                aic = np.nan
                bic = np.nan
        else:
            weight = np.nan
            r2 = np.nan
            rmse = np.nan
            aic = np.nan
            bic = np.nan

        record["prp_mix_cb_weight"] = weight
        record["prp_mix_r_squared"] = r2
        record["prp_mix_rmse"] = rmse
        record["prp_mix_aic"] = aic
        record["prp_mix_bic"] = bic

        results.append(record)

    return pd.DataFrame(results)


def load_or_compute_prp_bottleneck_mechanism_features(
    data_dir: Path | None = None,
    overwrite: bool = False,
    save: bool = True,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("prp")

    output_path = data_dir / MECHANISM_FILENAME
    if output_path.exists() and not overwrite:
        return pd.read_csv(output_path, encoding="utf-8-sig")

    features = compute_prp_bottleneck_features(data_dir=data_dir, **kwargs)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] PRP bottleneck features saved: {output_path}")
    return features

"""
Core helpers for preprocessing.
"""

from __future__ import annotations

import re
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .constants import (
    PARTICIPANT_ID_ALIASES,
    MALE_TOKENS_EXACT,
    FEMALE_TOKENS_EXACT,
    MALE_TOKENS_CONTAINS,
    FEMALE_TOKENS_CONTAINS,
)


def ensure_participant_id(df: pd.DataFrame, warn_threshold: float = 1.0) -> pd.DataFrame:
    """
    Ensure there is exactly one 'participant_id' column.
    Prefers an existing participant_id column, otherwise renames common aliases.
    """
    canonical = "participant_id"
    if canonical not in df.columns:
        for col in df.columns:
            if col in PARTICIPANT_ID_ALIASES and col != canonical:
                df = df.rename(columns={col: canonical})
                break
    if canonical not in df.columns:
        raise KeyError("No participant id column found in dataframe.")

    missing_count = df[canonical].isna().sum()
    missing_pct = missing_count / len(df) * 100 if len(df) > 0 else 0

    if missing_pct > warn_threshold:
        warnings.warn(
            f"participant_id column has {missing_pct:.1f}% missing values ({missing_count}/{len(df)} rows). "
            "This may cause silent data loss in downstream analyses.",
            UserWarning,
        )

    aliases = [col for col in df.columns if col in PARTICIPANT_ID_ALIASES and col != canonical]
    if aliases:
        df = df.drop(columns=aliases)
    return df


def _normalize_gender_string(value: object) -> str:
    if not isinstance(value, str):
        return ""
    cleaned = value.strip().lower()
    if not cleaned:
        return ""
    cleaned = re.sub(r"[^a-zㄱ-ㆎ가-힣]", "", cleaned)
    return cleaned


def normalize_gender_value(value: object) -> Optional[str]:
    """
    Normalize arbitrary gender text (Korean/English) to 'male'/'female'.
    Returns None if the value cannot be mapped.
    """
    token = _normalize_gender_string(value)
    if not token:
        return None
    if token in FEMALE_TOKENS_EXACT or any(t and t in token for t in FEMALE_TOKENS_CONTAINS):
        return "female"
    if token in MALE_TOKENS_EXACT or any(t and t in token for t in MALE_TOKENS_CONTAINS):
        return "male"
    return None


def normalize_gender_series(series: pd.Series) -> pd.Series:
    mapped = series.apply(normalize_gender_value)
    return pd.Series(mapped, index=series.index, dtype="object")


def coefficient_of_variation(series: pd.Series, min_n: int = 3) -> float:
    """Compute coefficient of variation (CV = std/mean)."""
    if len(series) < min_n or series.mean() == 0:
        return np.nan
    return series.std(ddof=1) / series.mean()


def median_absolute_deviation(series: pd.Series, min_n: int = 3) -> float:
    """Compute median absolute deviation (MAD)."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) < min_n:
        return np.nan
    median = np.median(values)
    return float(np.median(np.abs(values - median)))


def interquartile_range(series: pd.Series, min_n: int = 3) -> float:
    """Compute interquartile range (IQR = Q3 - Q1)."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) < min_n:
        return np.nan
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def skewness(series: pd.Series, min_n: int = 3) -> float:
    """Compute standardized third moment (skewness)."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) < min_n:
        return np.nan
    sd = float(np.std(values, ddof=1))
    if sd == 0:
        return np.nan
    return float(stats.skew(values, bias=False))


def mean_squared_successive_differences(series: pd.Series, min_n: int = 10) -> float:
    """Compute MSSD = mean((x_{t+1} - x_t)^2)."""
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) < min_n:
        return np.nan
    diffs = np.diff(values)
    if len(diffs) == 0:
        return np.nan
    return float(np.mean(diffs ** 2))


def run_lengths(mask: np.ndarray | pd.Series) -> list[int]:
    values = np.asarray(mask, dtype=bool)
    lengths: list[int] = []
    count = 0
    for val in values:
        if val:
            count += 1
        elif count:
            lengths.append(int(count))
            count = 0
    if count:
        lengths.append(int(count))
    return lengths


def run_length_stats(series: pd.Series, quantile: float = 0.5, min_n: int = 5) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) < min_n:
        return {
            "slow_run_mean": np.nan,
            "slow_run_max": np.nan,
            "fast_run_mean": np.nan,
            "fast_run_max": np.nan,
        }
    threshold = float(np.nanquantile(values, quantile))
    slow_mask = values >= threshold
    fast_mask = values < threshold
    slow_runs = run_lengths(slow_mask)
    fast_runs = run_lengths(fast_mask)
    return {
        "slow_run_mean": float(np.mean(slow_runs)) if slow_runs else 0.0,
        "slow_run_max": float(np.max(slow_runs)) if slow_runs else 0.0,
        "fast_run_mean": float(np.mean(fast_runs)) if fast_runs else 0.0,
        "fast_run_max": float(np.max(fast_runs)) if fast_runs else 0.0,
    }


def lag1_autocorrelation(series: pd.Series, min_n: int = 8, detrend: bool = True) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(values) < min_n:
        return np.nan
    if detrend:
        x = np.arange(len(values))
        try:
            slope, intercept = np.polyfit(x, values, 1)
            values = values - (slope * x + intercept)
        except Exception:
            return np.nan
    v1 = values[:-1]
    v2 = values[1:]
    if np.std(v1) == 0 or np.std(v2) == 0:
        return np.nan
    return float(np.corrcoef(v1, v2)[0, 1])


def dfa_alpha(
    series: pd.Series,
    min_n: int = 20,
    min_window: int = 4,
    n_scales: int = 6,
) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    n = len(values)
    if n < min_n:
        return np.nan

    y = np.cumsum(values - np.mean(values))
    max_window = n // 4
    if max_window < min_window:
        return np.nan

    sizes = np.unique(
        np.floor(
            np.logspace(np.log10(min_window), np.log10(max_window), n_scales)
        ).astype(int)
    )
    sizes = sizes[sizes >= 2]
    if len(sizes) < 2:
        return np.nan

    log_sizes = []
    log_fluct = []
    for size in sizes:
        n_segments = len(y) // size
        if n_segments < 2:
            continue
        trimmed = y[: n_segments * size]
        segments = trimmed.reshape(n_segments, size)
        rms_vals = []
        x = np.arange(size)
        for seg in segments:
            try:
                slope, intercept = np.polyfit(x, seg, 1)
            except Exception:
                continue
            trend = slope * x + intercept
            resid = seg - trend
            rms = np.sqrt(np.mean(resid ** 2))
            if np.isfinite(rms) and rms > 0:
                rms_vals.append(rms)
        if len(rms_vals) < 2:
            continue
        f_n = np.sqrt(np.mean(np.square(rms_vals)))
        if np.isfinite(f_n) and f_n > 0:
            log_sizes.append(np.log(size))
            log_fluct.append(np.log(f_n))

    if len(log_sizes) < 2:
        return np.nan

    slope, _ = np.polyfit(np.array(log_sizes), np.array(log_fluct), 1)
    return float(slope)


def compute_post_error_slowing(
    df: pd.DataFrame,
    rt_col: str,
    correct_col: str,
    participant_col: str = "participant_id",
    order_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute post-error slowing (PES) for each participant.

    PES = mean(RT after error) - mean(RT after correct)
    """
    records = []
    preferred_order_cols = []
    if order_col:
        preferred_order_cols.append(order_col)
    preferred_order_cols.extend(["trial_index", "trialIndex", "trial", "idx", "order"])

    for pid, grp in df.groupby(participant_col):
        grp = grp.copy()
        for candidate in preferred_order_cols:
            if candidate in grp.columns:
                grp = grp.sort_values(candidate)
                break
        else:
            grp = grp.sort_index()
        grp["prev_correct"] = grp[correct_col].shift(1)

        post_error = grp[grp["prev_correct"] == False]
        post_correct = grp[grp["prev_correct"] == True]

        post_error_mean = post_error[rt_col].mean() if len(post_error) > 0 else np.nan
        post_correct_mean = post_correct[rt_col].mean() if len(post_correct) > 0 else np.nan

        pes = (
            post_error_mean - post_correct_mean
            if pd.notna(post_error_mean) and pd.notna(post_correct_mean)
            else np.nan
        )

        records.append({
            participant_col: pid,
            "pes": pes,
            "post_error_rt": post_error_mean,
            "post_correct_rt": post_correct_mean,
        })

    return pd.DataFrame(records)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        mapped = series.astype(str).str.strip().str.lower().map(
            {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        )
        return mapped
    return series


def compute_error_cascade_metrics(
    correct_series: pd.Series,
    min_run: int = 2,
) -> dict[str, float]:
    correct = _coerce_bool_series(correct_series).dropna()
    if correct.empty:
        return {
            "error_cascade_count": np.nan,
            "error_cascade_rate": np.nan,
            "error_cascade_mean_len": np.nan,
            "error_cascade_max_len": np.nan,
            "error_cascade_trials": np.nan,
            "error_cascade_prop": np.nan,
        }
    errors = ~correct.astype(bool).to_numpy()
    runs = run_lengths(errors)
    cascades = [r for r in runs if r >= min_run]
    total_trials = len(errors)
    total_cascade = int(np.sum(cascades)) if cascades else 0
    return {
        "error_cascade_count": float(len(cascades)),
        "error_cascade_rate": float(len(cascades) / total_trials * 100) if total_trials else np.nan,
        "error_cascade_mean_len": float(np.mean(cascades)) if cascades else 0.0,
        "error_cascade_max_len": float(np.max(cascades)) if cascades else 0.0,
        "error_cascade_trials": float(total_cascade),
        "error_cascade_prop": float(total_cascade / total_trials * 100) if total_trials else np.nan,
    }


def compute_post_error_recovery_metrics(
    rt_series: pd.Series,
    correct_series: pd.Series,
    max_lag: int = 5,
    min_errors: int = 3,
) -> dict[str, float]:
    df = pd.DataFrame({
        "rt": pd.to_numeric(rt_series, errors="coerce"),
        "correct": _coerce_bool_series(correct_series),
    }).dropna(subset=["rt", "correct"])
    if df.empty or len(df) <= max_lag:
        base = {f"recovery_rt_lag{lag}": np.nan for lag in range(1, max_lag + 1)}
        base.update({f"recovery_acc_lag{lag}": np.nan for lag in range(1, max_lag + 1)})
        base.update({
            "recovery_rt_slope": np.nan,
            "recovery_acc_slope": np.nan,
            "recovery_n_errors": np.nan,
            "recovery_n_trials": len(df),
        })
        return base
    errors = ~df["correct"].astype(bool).to_numpy()
    error_idx = np.where(errors)[0]
    if len(error_idx) < min_errors:
        base = {f"recovery_rt_lag{lag}": np.nan for lag in range(1, max_lag + 1)}
        base.update({f"recovery_acc_lag{lag}": np.nan for lag in range(1, max_lag + 1)})
        base.update({
            "recovery_rt_slope": np.nan,
            "recovery_acc_slope": np.nan,
            "recovery_n_errors": float(len(error_idx)),
            "recovery_n_trials": len(df),
        })
        return base

    lag_rts: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}
    lag_accs: dict[int, list[float]] = {lag: [] for lag in range(1, max_lag + 1)}
    rts = df["rt"].to_numpy()
    correct = df["correct"].astype(bool).to_numpy()

    for idx in error_idx:
        for lag in range(1, max_lag + 1):
            nxt = idx + lag
            if nxt < len(rts):
                lag_rts[lag].append(float(rts[nxt]))
                lag_accs[lag].append(float(correct[nxt]))

    metrics: dict[str, float] = {
        "recovery_n_errors": float(len(error_idx)),
        "recovery_n_trials": float(len(df)),
    }
    rt_vals = []
    acc_vals = []
    lags = []
    for lag in range(1, max_lag + 1):
        rt_mean = float(np.mean(lag_rts[lag])) if lag_rts[lag] else np.nan
        acc_mean = float(np.mean(lag_accs[lag])) if lag_accs[lag] else np.nan
        metrics[f"recovery_rt_lag{lag}"] = rt_mean
        metrics[f"recovery_acc_lag{lag}"] = acc_mean
        if pd.notna(rt_mean):
            rt_vals.append(rt_mean)
            lags.append(lag)
        if pd.notna(acc_mean):
            acc_vals.append(acc_mean)

    if len(rt_vals) >= 3:
        metrics["recovery_rt_slope"] = float(np.polyfit(lags, rt_vals, 1)[0])
    else:
        metrics["recovery_rt_slope"] = np.nan

    if len(acc_vals) >= 3:
        metrics["recovery_acc_slope"] = float(np.polyfit(range(1, len(acc_vals) + 1), acc_vals, 1)[0])
    else:
        metrics["recovery_acc_slope"] = np.nan

    return metrics


def compute_momentum_metrics(
    rt_series: pd.Series,
    correct_series: pd.Series,
    max_streak: int = 5,
    min_per_streak: int = 5,
) -> dict[str, float]:
    df = pd.DataFrame({
        "rt": pd.to_numeric(rt_series, errors="coerce"),
        "correct": _coerce_bool_series(correct_series),
    }).dropna(subset=["rt", "correct"])
    if df.empty:
        return {
            "momentum_slope": np.nan,
            "momentum_mean_streak": np.nan,
            "momentum_max_streak": np.nan,
            **{f"momentum_rt_streak_{s}": np.nan for s in range(max_streak + 1)},
        }

    correct = df["correct"].astype(bool).to_numpy()
    rts = df["rt"].to_numpy()
    streaks: list[int] = []
    current = 0
    for c in correct:
        if c:
            current += 1
        else:
            current = 0
        streaks.append(current)

    streaks_arr = np.array(streaks)
    streak_rt: dict[int, float] = {}
    for streak in range(max_streak + 1):
        mask = streaks_arr == streak
        if mask.sum() >= min_per_streak:
            streak_rt[streak] = float(np.nanmean(rts[mask]))

    result = {
        "momentum_mean_streak": float(np.mean(streaks_arr)),
        "momentum_max_streak": float(np.max(streaks_arr)),
        **{f"momentum_rt_streak_{s}": streak_rt.get(s, np.nan) for s in range(max_streak + 1)},
    }

    valid = [(s, rt) for s, rt in streak_rt.items() if np.isfinite(rt)]
    if len(valid) >= 3:
        x = np.array([v[0] for v in valid], dtype=float)
        y = np.array([v[1] for v in valid], dtype=float)
        slope, _, _, _, _ = stats.linregress(x, y)
        result["momentum_slope"] = float(slope)
    else:
        result["momentum_slope"] = np.nan

    return result


def compute_volatility_metrics(
    rt_series: pd.Series,
    min_n: int = 10,
) -> dict[str, float]:
    rts = pd.to_numeric(rt_series, errors="coerce").dropna().to_numpy()
    if len(rts) < min_n:
        return {"volatility_rmssd": np.nan, "volatility_adj": np.nan}
    diffs = np.diff(rts)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else np.nan
    x = np.arange(len(rts), dtype=float)
    try:
        slope, intercept = np.polyfit(x, rts, 1)
        residuals = rts - (slope * x + intercept)
        adj = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan
    except Exception:
        adj = np.nan
    return {"volatility_rmssd": rmssd, "volatility_adj": adj}


def compute_iiv_parameters(
    rt_series: pd.Series,
    min_n: int = 10,
) -> dict[str, float]:
    rts = pd.to_numeric(rt_series, errors="coerce").dropna().to_numpy()
    n_trials = len(rts)
    if n_trials < min_n:
        return {
            "iiv_intercept": np.nan,
            "iiv_slope": np.nan,
            "iiv_slope_p": np.nan,
            "iiv_residual_sd": np.nan,
            "iiv_raw_cv": np.nan,
            "iiv_n_trials": float(n_trials),
            "iiv_r_squared": np.nan,
        }
    x = np.linspace(0.0, 1.0, n_trials)
    slope, intercept, r_value, p_value, _ = stats.linregress(x, rts)
    predicted = intercept + slope * x
    residual_sd = float(np.std(rts - predicted, ddof=2)) if n_trials > 2 else np.nan
    raw_cv = float(np.std(rts, ddof=1) / np.mean(rts)) if np.mean(rts) > 0 else np.nan
    return {
        "iiv_intercept": float(intercept),
        "iiv_slope": float(slope),
        "iiv_slope_p": float(p_value),
        "iiv_residual_sd": residual_sd,
        "iiv_raw_cv": raw_cv,
        "iiv_n_trials": float(n_trials),
        "iiv_r_squared": float(r_value ** 2),
    }


def fit_exgaussian_moments(rts: np.ndarray, min_n: int = 20) -> tuple[float, float, float]:
    values = np.asarray(rts, dtype=float)
    values = values[(values > 0) & np.isfinite(values)]
    if len(values) < min_n:
        return np.nan, np.nan, np.nan
    m1 = float(np.mean(values))
    m2 = float(np.var(values))
    m3 = stats.moment(values, moment=3)
    if not np.isfinite(m3) or m3 <= 0:
        return m1, float(np.sqrt(m2)), 0.0
    tau = float((m3 / 2) ** (1 / 3))
    mu = m1 - tau
    sigma_sq = m2 - tau ** 2
    if sigma_sq <= 0:
        sigma = float(np.sqrt(m2) * 0.5)
    else:
        sigma = float(np.sqrt(sigma_sq))
    return mu, sigma, tau


def compute_tau_quartile_metrics(
    rt_series: pd.Series,
    correct_series: pd.Series | None = None,
    n_quartiles: int = 4,
    min_n: int = 40,
    min_per_quartile: int = 10,
) -> dict[str, float]:
    df = pd.DataFrame({"rt": pd.to_numeric(rt_series, errors="coerce")})
    if correct_series is not None:
        df["correct"] = _coerce_bool_series(correct_series)
        df = df.dropna(subset=["rt", "correct"])
        df = df[df["correct"].astype(bool)]
    else:
        df = df.dropna(subset=["rt"])
    if len(df) < min_n:
        return {
            "tau_q1": np.nan,
            "tau_q2": np.nan,
            "tau_q3": np.nan,
            "tau_q4": np.nan,
            "tau_slope": np.nan,
        }
    df = df.reset_index(drop=True)
    df["quartile"] = pd.qcut(range(len(df)), q=n_quartiles, labels=list(range(1, n_quartiles + 1)))
    taus: dict[int, float] = {}
    for q in range(1, n_quartiles + 1):
        q_vals = df[df["quartile"] == q]["rt"].to_numpy()
        if len(q_vals) >= min_per_quartile:
            _, _, tau = fit_exgaussian_moments(q_vals)
            taus[q] = float(tau)
    if len(taus) < n_quartiles:
        return {
            "tau_q1": np.nan,
            "tau_q2": np.nan,
            "tau_q3": np.nan,
            "tau_q4": np.nan,
            "tau_slope": np.nan,
        }
    tau_q1 = taus.get(1, np.nan)
    tau_q4 = taus.get(n_quartiles, np.nan)
    tau_slope = float(tau_q4 - tau_q1) if pd.notna(tau_q1) and pd.notna(tau_q4) else np.nan
    return {
        "tau_q1": tau_q1,
        "tau_q2": taus.get(2, np.nan),
        "tau_q3": taus.get(3, np.nan),
        "tau_q4": tau_q4,
        "tau_slope": tau_slope,
    }


def compute_fatigue_slopes(
    rt_series: pd.Series,
    correct_series: pd.Series | None = None,
    n_quartiles: int = 4,
    min_n: int = 20,
    min_per_quartile: int = 5,
) -> dict[str, float]:
    df = pd.DataFrame({"rt": pd.to_numeric(rt_series, errors="coerce")})
    if correct_series is not None:
        df["correct"] = _coerce_bool_series(correct_series)
    df = df.dropna(subset=["rt"])
    if len(df) < min_n:
        return {
            "rt_fatigue_slope": np.nan,
            "cv_fatigue_slope": np.nan,
            "acc_fatigue_slope": np.nan,
        }
    df = df.reset_index(drop=True)
    df["quartile"] = pd.qcut(range(len(df)), q=n_quartiles, labels=list(range(1, n_quartiles + 1)))

    def _quartile_metrics(q: int) -> dict[str, float]:
        q_df = df[df["quartile"] == q]
        if len(q_df) < min_per_quartile:
            return {"mean_rt": np.nan, "cv_rt": np.nan, "acc": np.nan}
        mean_rt = float(q_df["rt"].mean())
        cv_rt = float(q_df["rt"].std(ddof=1) / mean_rt) if mean_rt > 0 else np.nan
        if "correct" in q_df.columns and q_df["correct"].notna().any():
            acc = float(q_df["correct"].astype(bool).mean())
        else:
            acc = np.nan
        return {"mean_rt": mean_rt, "cv_rt": cv_rt, "acc": acc}

    q1 = _quartile_metrics(1)
    q4 = _quartile_metrics(n_quartiles)
    return {
        "rt_fatigue_slope": float(q4["mean_rt"] - q1["mean_rt"]) if pd.notna(q4["mean_rt"]) and pd.notna(q1["mean_rt"]) else np.nan,
        "cv_fatigue_slope": float(q4["cv_rt"] - q1["cv_rt"]) if pd.notna(q4["cv_rt"]) and pd.notna(q1["cv_rt"]) else np.nan,
        "acc_fatigue_slope": float(q4["acc"] - q1["acc"]) if pd.notna(q4["acc"]) and pd.notna(q1["acc"]) else np.nan,
    }


def compute_error_awareness_metrics(
    rt_series: pd.Series,
    correct_series: pd.Series,
    min_post_error: int = 3,
    min_post_correct: int = 10,
) -> dict[str, float]:
    df = pd.DataFrame({
        "rt": pd.to_numeric(rt_series, errors="coerce"),
        "correct": _coerce_bool_series(correct_series),
    }).dropna(subset=["rt", "correct"])
    if df.empty:
        return {
            "post_error_cv": np.nan,
            "post_correct_cv": np.nan,
            "post_error_cv_reduction": np.nan,
            "post_error_acc": np.nan,
            "post_correct_acc": np.nan,
            "post_error_acc_diff": np.nan,
            "post_error_recovery_rate": np.nan,
            "pes_adaptive": np.nan,
            "pes_maladaptive": np.nan,
        }
    df["prev_correct"] = df["correct"].shift(1)
    df["next_correct"] = df["correct"].shift(-1)

    post_error = df[df["prev_correct"] == False]
    post_correct = df[df["prev_correct"] == True]
    if len(post_error) < min_post_error or len(post_correct) < min_post_correct:
        return {
            "post_error_cv": np.nan,
            "post_correct_cv": np.nan,
            "post_error_cv_reduction": np.nan,
            "post_error_acc": np.nan,
            "post_correct_acc": np.nan,
            "post_error_acc_diff": np.nan,
            "post_error_recovery_rate": np.nan,
            "pes_adaptive": np.nan,
            "pes_maladaptive": np.nan,
        }

    post_error_mean = float(post_error["rt"].mean())
    post_correct_mean = float(post_correct["rt"].mean())
    post_error_cv = float(post_error["rt"].std(ddof=1) / post_error_mean) if post_error_mean > 0 else np.nan
    post_correct_cv = float(post_correct["rt"].std(ddof=1) / post_correct_mean) if post_correct_mean > 0 else np.nan
    cv_reduction = float(post_correct_cv - post_error_cv) if pd.notna(post_correct_cv) and pd.notna(post_error_cv) else np.nan

    post_error_acc = float(post_error["correct"].astype(bool).mean())
    post_correct_acc = float(post_correct["correct"].astype(bool).mean())
    acc_diff = float(post_error_acc - post_correct_acc)

    error_trials = df[df["correct"] == False]
    if len(error_trials) >= min_post_error:
        recovery_rate = float(error_trials["next_correct"].astype(bool).mean())
    else:
        recovery_rate = np.nan

    pes = post_error_mean - post_correct_mean if pd.notna(post_error_mean) and pd.notna(post_correct_mean) else np.nan
    adaptive = 1.0 if pd.notna(pes) and pes > 0 and acc_diff >= 0 else 0.0
    maladaptive = 1.0 if pd.notna(pes) and pes > 0 and acc_diff < 0 else 0.0

    return {
        "post_error_cv": post_error_cv,
        "post_correct_cv": post_correct_cv,
        "post_error_cv_reduction": cv_reduction,
        "post_error_acc": post_error_acc,
        "post_correct_acc": post_correct_acc,
        "post_error_acc_diff": acc_diff,
        "post_error_recovery_rate": recovery_rate,
        "pes_adaptive": adaptive,
        "pes_maladaptive": maladaptive,
    }


def compute_pre_error_slope_metrics(
    rt_series: pd.Series,
    correct_series: pd.Series,
    window: int = 3,
    min_errors: int = 3,
) -> dict[str, float]:
    df = pd.DataFrame({
        "rt": pd.to_numeric(rt_series, errors="coerce"),
        "correct": _coerce_bool_series(correct_series),
    }).dropna(subset=["rt", "correct"])
    if df.empty or len(df) <= window:
        return {
            "pre_error_slope_mean": np.nan,
            "pre_error_slope_std": np.nan,
            "pre_error_n": np.nan,
        }

    errors = ~df["correct"].astype(bool).to_numpy()
    rts = df["rt"].to_numpy()
    error_idx = np.where(errors)[0]
    slopes: list[float] = []
    positions = np.arange(-window, 0, 1)

    for idx in error_idx:
        if idx < window:
            continue
        pre_rts = rts[idx - window:idx]
        if len(pre_rts) < window or np.any(~np.isfinite(pre_rts)):
            continue
        try:
            slope = np.polyfit(positions, pre_rts, 1)[0]
            slopes.append(float(slope))
        except Exception:
            continue

    if len(slopes) < min_errors:
        return {
            "pre_error_slope_mean": np.nan,
            "pre_error_slope_std": np.nan,
            "pre_error_n": float(len(slopes)),
        }

    return {
        "pre_error_slope_mean": float(np.mean(slopes)),
        "pre_error_slope_std": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else 0.0,
        "pre_error_n": float(len(slopes)),
    }


def compute_speed_accuracy_metrics(
    rt_series: pd.Series,
    correct_series: pd.Series,
    min_n: int = 10,
) -> dict[str, float]:
    df = pd.DataFrame({
        "rt": pd.to_numeric(rt_series, errors="coerce"),
        "correct": _coerce_bool_series(correct_series),
    }).dropna(subset=["rt", "correct"])
    df = df[df["rt"] > 0]
    if len(df) < min_n:
        return {
            "mean_rt": np.nan,
            "accuracy": np.nan,
            "error_rate": np.nan,
            "ies": np.nan,
            "n_trials": float(len(df)),
        }

    mean_rt = float(df["rt"].mean())
    accuracy = float(df["correct"].astype(bool).mean())
    error_rate = float(1.0 - accuracy)
    ies = float(mean_rt / accuracy) if accuracy > 0 else np.nan
    return {
        "mean_rt": mean_rt,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "ies": ies,
        "n_trials": float(len(df)),
    }

"""PRP bottleneck shape feature derivation (linear/exponential)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from ..constants import get_results_dir
from ..core import ensure_participant_id

MECHANISM_FILENAME = "5_prp_bottleneck_shape_features.csv"


def _exp_decay(x: np.ndarray, amplitude: float, decay: float, asymptote: float) -> np.ndarray:
    return amplitude * np.exp(-decay * x) + asymptote


def _extract_soa_columns(df: pd.DataFrame) -> Dict[int, str]:
    pattern = re.compile(r"^(rt2|rt)_soa_(\d+)$", re.IGNORECASE)
    soa_cols: Dict[int, str] = {}
    for col in df.columns:
        match = pattern.match(col)
        if not match:
            continue
        soa = int(match.group(2))
        if soa in soa_cols:
            # Prefer rt2_soa if both exist.
            if col.lower().startswith("rt2"):
                soa_cols[soa] = col
        else:
            soa_cols[soa] = col
    return dict(sorted(soa_cols.items()))


def _fit_linear(soas: np.ndarray, rts: np.ndarray) -> Dict[str, float]:
    if len(soas) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan}
    slope, intercept, r_value, _, _ = stats.linregress(soas, rts)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r_value ** 2),
    }


def _fit_exponential(soas: np.ndarray, rts: np.ndarray) -> Dict[str, float]:
    if len(soas) < 3:
        return {"amplitude": np.nan, "decay_rate": np.nan, "asymptote": np.nan, "r2": np.nan}
    amplitude_init = float(max(rts) - min(rts))
    asymptote_init = float(min(rts))
    decay_init = 0.002
    try:
        params, _ = curve_fit(
            _exp_decay,
            soas,
            rts,
            p0=[amplitude_init, decay_init, asymptote_init],
            bounds=([0.0, 0.0, 0.0], [5000.0, 0.5, 5000.0]),
            maxfev=5000,
        )
        amplitude, decay_rate, asymptote = params
        preds = _exp_decay(soas, amplitude, decay_rate, asymptote)
        ss_res = float(np.sum((rts - preds) ** 2))
        ss_tot = float(np.sum((rts - np.mean(rts)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {
            "amplitude": float(amplitude),
            "decay_rate": float(decay_rate),
            "asymptote": float(asymptote),
            "r2": float(r2),
        }
    except Exception:
        return {"amplitude": np.nan, "decay_rate": np.nan, "asymptote": np.nan, "r2": np.nan}


def compute_prp_bottleneck_shape_features(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    if data_dir is None:
        data_dir = get_results_dir("prp")

    summary_path = Path(data_dir) / "3_cognitive_tests_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame()

    summary = pd.read_csv(summary_path, encoding="utf-8")
    summary = ensure_participant_id(summary)
    if "testName" in summary.columns:
        summary["testName"] = summary["testName"].astype(str).str.lower()
        summary = summary[summary["testName"] == "prp"].copy()

    if summary.empty:
        return pd.DataFrame()

    soa_cols = _extract_soa_columns(summary)
    if len(soa_cols) < 3:
        return pd.DataFrame()

    results = []
    for _, row in summary.iterrows():
        pid = row["participant_id"]
        soas = []
        rts = []
        for soa, col in soa_cols.items():
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                soas.append(float(soa))
                rts.append(float(val))
        if len(soas) < 3:
            results.append({
                "participant_id": pid,
                "prp_shape_linear_slope": np.nan,
                "prp_shape_linear_intercept": np.nan,
                "prp_shape_linear_r2": np.nan,
                "prp_shape_exp_amplitude": np.nan,
                "prp_shape_exp_decay_rate": np.nan,
                "prp_shape_exp_asymptote": np.nan,
                "prp_shape_exp_r2": np.nan,
                "prp_shape_recovery_half_life": np.nan,
                "prp_shape_bottleneck_traditional": np.nan,
                "prp_shape_exp_better_fit": np.nan,
                "prp_shape_n_soa": float(len(soas)),
            })
            continue

        soas_arr = np.array(soas, dtype=float)
        rts_arr = np.array(rts, dtype=float)
        order = np.argsort(soas_arr)
        soas_arr = soas_arr[order]
        rts_arr = rts_arr[order]

        linear = _fit_linear(soas_arr, rts_arr)
        exp = _fit_exponential(soas_arr, rts_arr)

        half_life = np.log(2) / exp["decay_rate"] if pd.notna(exp["decay_rate"]) and exp["decay_rate"] > 0 else np.nan
        bottleneck_traditional = float(rts_arr[0] - rts_arr[-1]) if len(rts_arr) >= 2 else np.nan

        if pd.notna(exp["r2"]) and pd.notna(linear["r2"]):
            exp_better = float(exp["r2"] > linear["r2"])
        else:
            exp_better = np.nan

        results.append({
            "participant_id": pid,
            "prp_shape_linear_slope": linear["slope"],
            "prp_shape_linear_intercept": linear["intercept"],
            "prp_shape_linear_r2": linear["r2"],
            "prp_shape_exp_amplitude": exp["amplitude"],
            "prp_shape_exp_decay_rate": exp["decay_rate"],
            "prp_shape_exp_asymptote": exp["asymptote"],
            "prp_shape_exp_r2": exp["r2"],
            "prp_shape_recovery_half_life": float(half_life) if pd.notna(half_life) else np.nan,
            "prp_shape_bottleneck_traditional": bottleneck_traditional,
            "prp_shape_exp_better_fit": exp_better,
            "prp_shape_n_soa": float(len(soas_arr)),
        })

    return pd.DataFrame(results)


def load_or_compute_prp_bottleneck_shape_features(
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

    features = compute_prp_bottleneck_shape_features(data_dir=data_dir)
    if save and not features.empty:
        features.to_csv(output_path, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"[OK] PRP bottleneck shape features saved: {output_path}")
    return features

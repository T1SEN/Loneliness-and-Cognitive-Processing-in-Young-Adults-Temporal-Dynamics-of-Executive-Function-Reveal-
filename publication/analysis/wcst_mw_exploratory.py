"""
Exploratory WCST analyses for mind-wandering/DMN proxies (no FDR correction).

Outputs:
- wcst_exgaussian_features.csv
- wcst_exgaussian_regression.csv
- wcst_error_cascade_regression.csv
- wcst_lapse_error_overlap.csv
- wcst_lapse_error_overlap_regression.csv
- wcst_mediation_results.csv
- wcst_mw_exploratory_summary.txt
"""

from __future__ import annotations

import sys
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import minimize
from scipy.stats import exponnorm, norm

from publication.analysis.utils import get_analysis_data, get_output_dir
from publication.preprocessing.constants import WCST_RT_MAX
from publication.preprocessing.wcst.loaders import load_wcst_trials

try:
    from hmmlearn import hmm as hmm_module
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


def _fit_exgaussian(rts: np.ndarray, min_trials: int = 20) -> dict[str, float]:
    rts = np.asarray(rts, dtype=float)
    rts = rts[np.isfinite(rts)]
    if len(rts) < min_trials:
        return {"mu": np.nan, "sigma": np.nan, "tau": np.nan}

    mean_rt = float(np.mean(rts))
    std_rt = float(np.std(rts))
    if std_rt <= 0:
        return {"mu": np.nan, "sigma": np.nan, "tau": np.nan}

    skew = float(np.mean(((rts - mean_rt) / std_rt) ** 3))
    tau_init = max(10.0, (abs(skew) / 2.0) ** (1.0 / 3.0) * std_rt)
    mu_init = max(100.0, mean_rt - tau_init)
    sigma_init = max(10.0, np.sqrt(max(0.0, std_rt ** 2 - tau_init ** 2)))

    def neg_loglik(params: np.ndarray) -> float:
        mu, sigma, tau = params
        if sigma <= 0 or tau <= 0:
            return 1e10
        k = tau / sigma
        try:
            return -np.sum(exponnorm.logpdf(rts, k, loc=mu, scale=sigma))
        except Exception:
            return 1e10

    result = minimize(
        neg_loglik,
        x0=[mu_init, sigma_init, tau_init],
        bounds=[(100.0, WCST_RT_MAX), (10.0, 2000.0), (10.0, 5000.0)],
        method="L-BFGS-B",
    )
    if not result.success:
        return {"mu": np.nan, "sigma": np.nan, "tau": np.nan}

    mu, sigma, tau = result.x
    return {"mu": float(mu), "sigma": float(sigma), "tau": float(tau)}


def compute_wcst_exgaussian_features(
    trials: pd.DataFrame,
    min_trials: int = 20,
) -> pd.DataFrame:
    trials = trials.copy()
    trials["rt_ms"] = pd.to_numeric(trials["rt_ms"], errors="coerce")
    trials = trials[trials["rt_ms"].notna()]

    results: list[dict[str, float]] = []
    for pid, group in trials.groupby("participant_id"):
        record: dict[str, float] = {"participant_id": pid}
        group_all = group
        if "correct" in group_all.columns:
            group_correct = group_all[group_all["correct"] == True].copy()
        else:
            group_correct = group_all.copy()

        params_all = _fit_exgaussian(group_all["rt_ms"].values, min_trials=min_trials)
        record["wcst_exg_mu"] = params_all["mu"]
        record["wcst_exg_sigma"] = params_all["sigma"]
        record["wcst_exg_tau"] = params_all["tau"]

        params_correct = _fit_exgaussian(group_correct["rt_ms"].values, min_trials=min_trials)
        record["wcst_exg_correct_mu"] = params_correct["mu"]
        record["wcst_exg_correct_sigma"] = params_correct["sigma"]
        record["wcst_exg_correct_tau"] = params_correct["tau"]

        results.append(record)

    return pd.DataFrame(results)


def _mark_error_cascade(errors: np.ndarray, min_run: int = 2) -> np.ndarray:
    mask = np.zeros(len(errors), dtype=bool)
    run = 0
    for idx, val in enumerate(errors):
        if val:
            run += 1
        else:
            if run >= min_run:
                mask[idx - run:idx] = True
            run = 0
    if run >= min_run:
        mask[len(errors) - run:] = True
    return mask


def compute_lapse_error_overlap(
    trials: pd.DataFrame,
    min_trials: int = 50,
    min_run: int = 2,
) -> pd.DataFrame:
    if not HMM_AVAILABLE:
        return pd.DataFrame()

    results: list[dict[str, float]] = []
    trials = trials.sort_values(["participant_id", "trial_index"]) if "trial_index" in trials.columns else trials

    for pid, pdata in trials.groupby("participant_id"):
        if len(pdata) < min_trials:
            continue

        rts = pd.to_numeric(pdata["rt_ms"], errors="coerce").to_numpy()
        valid_mask = np.isfinite(rts)
        rts_valid = rts[valid_mask]
        if len(rts_valid) < min_trials:
            continue

        try:
            model = hmm_module.GaussianHMM(
                n_components=2,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(rts_valid.reshape(-1, 1))
            states = model.predict(rts_valid.reshape(-1, 1))
            means = model.means_.flatten()
            lapse_state = int(np.argmax(means))
            lapse_mask = states == lapse_state
        except Exception:
            continue

        if "correct" not in pdata.columns:
            continue
        correct = pdata["correct"].astype(bool).to_numpy()
        correct = correct[valid_mask]
        errors = ~correct
        cascade_mask = _mark_error_cascade(errors, min_run=min_run)

        cascade_trials = int(cascade_mask.sum())
        lapse_trials = int(np.sum(lapse_mask))
        overlap_count = int(np.sum(lapse_mask & cascade_mask))

        overlap_prop_cascade = (
            overlap_count / cascade_trials if cascade_trials > 0 else np.nan
        )
        overlap_prop_lapse = (
            overlap_count / lapse_trials if lapse_trials > 0 else np.nan
        )

        results.append({
            "participant_id": pid,
            "wcst_error_cascade_trials": float(cascade_trials),
            "wcst_lapse_trials": float(lapse_trials),
            "wcst_lapse_error_overlap_trials": float(overlap_count),
            "wcst_lapse_error_overlap_prop_cascade": float(overlap_prop_cascade),
            "wcst_lapse_error_overlap_prop_lapse": float(overlap_prop_lapse),
        })

    return pd.DataFrame(results)


def run_ucla_regression(
    df: pd.DataFrame,
    outcome: str,
    min_n: int = 30,
) -> dict | None:
    required = [
        outcome,
        "z_ucla_score",
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
    ]
    cols = [c for c in required if c in df.columns]
    data = df[cols].dropna()
    if len(data) < min_n:
        return None

    formula = (
        f"{outcome} ~ z_ucla_score + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + C(gender_male)"
    )
    try:
        model = smf.ols(formula, data=data).fit()
    except Exception:
        return None

    return {
        "outcome_column": outcome,
        "n": int(len(data)),
        "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
        "ucla_t": float(model.tvalues.get("z_ucla_score", np.nan)),
        "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        "r2": float(model.rsquared),
        "adj_r2": float(model.rsquared_adj),
        "cov_type": "OLS",
    }


def _sobel_p(a: float, b: float, se_a: float, se_b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or se_a <= 0 or se_b <= 0:
        return np.nan
    sobel_se = np.sqrt((b ** 2) * (se_a ** 2) + (a ** 2) * (se_b ** 2))
    if sobel_se <= 0:
        return np.nan
    z = (a * b) / sobel_se
    return float(2 * (1 - norm.cdf(abs(z))))


def bootstrap_mediation(
    df: pd.DataFrame,
    x: str,
    m: str,
    y: str,
    covariates: list[str],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    cols = [x, m, y] + covariates
    data = df[cols].dropna()
    n = len(data)
    if n < 30:
        return {
            "n": n,
            "a": np.nan,
            "b": np.nan,
            "c_prime": np.nan,
            "indirect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_boot": np.nan,
            "sobel_p": np.nan,
        }

    cov_terms = [f"C({c})" if c == "gender_male" else c for c in covariates]
    stage1_formula = f"{m} ~ {x} + " + " + ".join(cov_terms)
    stage2_formula = f"{y} ~ {x} + {m} + " + " + ".join(cov_terms)

    stage1 = smf.ols(stage1_formula, data=data).fit()
    stage2 = smf.ols(stage2_formula, data=data).fit()

    a = float(stage1.params.get(x, np.nan))
    b = float(stage2.params.get(m, np.nan))
    c_prime = float(stage2.params.get(x, np.nan))
    indirect = a * b

    rng = np.random.default_rng(seed)
    boot_vals: list[float] = []
    for _ in range(n_boot):
        sample = data.sample(n=n, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        try:
            s1 = smf.ols(stage1_formula, data=sample).fit()
            s2 = smf.ols(stage2_formula, data=sample).fit()
            a_b = s1.params.get(x, np.nan) * s2.params.get(m, np.nan)
            if np.isfinite(a_b):
                boot_vals.append(float(a_b))
        except Exception:
            continue

    if len(boot_vals) == 0:
        ci_low = np.nan
        ci_high = np.nan
        p_boot = np.nan
    else:
        boot_arr = np.array(boot_vals)
        ci_low = float(np.percentile(boot_arr, 2.5))
        ci_high = float(np.percentile(boot_arr, 97.5))
        pos = float(np.mean(boot_arr > 0))
        neg = float(np.mean(boot_arr < 0))
        p_boot = float(2 * min(pos, neg))

    sobel_p = _sobel_p(
        a,
        b,
        float(stage1.bse.get(x, np.nan)),
        float(stage2.bse.get(m, np.nan)),
    )

    return {
        "n": n,
        "a": a,
        "b": b,
        "c_prime": c_prime,
        "indirect": indirect,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_boot": p_boot,
        "sobel_p": sobel_p,
    }


def run() -> None:
    output_dir = get_output_dir("wcst")
    trials, _ = load_wcst_trials(apply_trial_filters=True, filter_rt=True)

    exg = compute_wcst_exgaussian_features(trials)
    exg_path = output_dir / "wcst_exgaussian_features.csv"
    exg.to_csv(exg_path, index=False, encoding="utf-8-sig")

    master = get_analysis_data("wcst")
    merged = master.merge(exg, on="participant_id", how="left")

    exg_metrics = [
        "wcst_exg_mu",
        "wcst_exg_sigma",
        "wcst_exg_tau",
        "wcst_exg_correct_mu",
        "wcst_exg_correct_sigma",
        "wcst_exg_correct_tau",
    ]
    exg_results = []
    for col in exg_metrics:
        if col not in merged.columns:
            continue
        res = run_ucla_regression(merged, col)
        if res:
            res["outcome"] = col
            exg_results.append(res)
    exg_df = pd.DataFrame(exg_results)
    exg_reg_path = output_dir / "wcst_exgaussian_regression.csv"
    exg_df.to_csv(exg_reg_path, index=False, encoding="utf-8-sig")

    error_metrics = [
        "wcst_error_cascade_count",
        "wcst_error_cascade_rate",
        "wcst_error_cascade_mean_len",
        "wcst_error_cascade_max_len",
        "wcst_error_cascade_trials",
        "wcst_error_cascade_prop",
    ]
    error_results = []
    for col in error_metrics:
        if col not in master.columns:
            continue
        res = run_ucla_regression(master, col)
        if res:
            res["outcome"] = col
            error_results.append(res)
    error_df = pd.DataFrame(error_results)
    error_path = output_dir / "wcst_error_cascade_regression.csv"
    error_df.to_csv(error_path, index=False, encoding="utf-8-sig")

    overlap = compute_lapse_error_overlap(trials)
    overlap_path = output_dir / "wcst_lapse_error_overlap.csv"
    if not overlap.empty:
        overlap.to_csv(overlap_path, index=False, encoding="utf-8-sig")
        overlap_merged = master.merge(overlap, on="participant_id", how="left")
        overlap_metrics = [
            "wcst_lapse_error_overlap_prop_cascade",
            "wcst_lapse_error_overlap_prop_lapse",
        ]
        overlap_results = []
        for col in overlap_metrics:
            if col not in overlap_merged.columns:
                continue
            res = run_ucla_regression(overlap_merged, col)
            if res:
                res["outcome"] = col
                overlap_results.append(res)
        overlap_df = pd.DataFrame(overlap_results)
    else:
        overlap_df = pd.DataFrame()
    overlap_reg_path = output_dir / "wcst_lapse_error_overlap_regression.csv"
    overlap_df.to_csv(overlap_reg_path, index=False, encoding="utf-8-sig")

    covariates = [
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
    ]
    mediation_specs = [
        ("wcst_hmm_lapse_occupancy", "wcst_error_cascade_rate"),
        ("wcst_hmm_stay_lapse", "wcst_post_switch_error_rate"),
        ("wcst_slow_prob_shift_k0", "wcst_trials_to_first_conceptual_resp"),
        ("wcst_exg_tau", "wcst_error_cascade_rate"),
    ]

    mediation_rows = []
    for mediator, outcome in mediation_specs:
        if mediator not in merged.columns or outcome not in merged.columns:
            continue
        res = bootstrap_mediation(
            merged,
            x="z_ucla_score",
            m=mediator,
            y=outcome,
            covariates=covariates,
            n_boot=1000,
        )
        res.update({
            "x": "z_ucla_score",
            "mediator": mediator,
            "outcome": outcome,
        })
        mediation_rows.append(res)
    mediation_df = pd.DataFrame(mediation_rows)
    mediation_path = output_dir / "wcst_mediation_results.csv"
    mediation_df.to_csv(mediation_path, index=False, encoding="utf-8-sig")

    summary_lines: list[str] = []
    summary_lines.append("WCST MW Exploratory Summary (uncorrected)")
    summary_lines.append("=" * 72)
    summary_lines.append("")
    summary_lines.append(f"Ex-Gaussian features: {len(exg_df)} models")
    if not exg_df.empty:
        sig_exg = exg_df[exg_df["ucla_p"] < 0.05]
        summary_lines.append(f"  p<0.05: {len(sig_exg)}")
        for _, row in sig_exg.iterrows():
            summary_lines.append(
                f"  - {row['outcome']} | beta={row['ucla_beta']:.4g} | p={row['ucla_p']:.4g}"
            )
    summary_lines.append("")
    summary_lines.append(f"Error cascade metrics: {len(error_df)} models")
    if not error_df.empty:
        sig_err = error_df[error_df["ucla_p"] < 0.05]
        summary_lines.append(f"  p<0.05: {len(sig_err)}")
        for _, row in sig_err.iterrows():
            summary_lines.append(
                f"  - {row['outcome']} | beta={row['ucla_beta']:.4g} | p={row['ucla_p']:.4g}"
            )
    summary_lines.append("")
    summary_lines.append(f"Lapse-error overlap metrics: {len(overlap_df)} models")
    if not overlap_df.empty:
        sig_overlap = overlap_df[overlap_df["ucla_p"] < 0.05]
        summary_lines.append(f"  p<0.05: {len(sig_overlap)}")
        for _, row in sig_overlap.iterrows():
            summary_lines.append(
                f"  - {row['outcome']} | beta={row['ucla_beta']:.4g} | p={row['ucla_p']:.4g}"
            )
    summary_lines.append("")
    summary_lines.append(f"Mediation models: {len(mediation_df)}")
    for _, row in mediation_df.iterrows():
        summary_lines.append(
            f"  - M={row['mediator']} -> Y={row['outcome']} | indirect={row['indirect']:.4g} "
            f"CI[{row['ci_low']:.4g}, {row['ci_high']:.4g}] p_boot={row['p_boot']:.4g}"
        )

    summary_path = output_dir / "wcst_mw_exploratory_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[OK] Ex-Gaussian features: {exg_path}")
    print(f"[OK] Ex-Gaussian regression: {exg_reg_path}")
    print(f"[OK] Error cascade regression: {error_path}")
    print(f"[OK] Lapse-error overlap: {overlap_path}")
    print(f"[OK] Lapse-error overlap regression: {overlap_reg_path}")
    print(f"[OK] Mediation: {mediation_path}")
    print(f"[OK] Summary: {summary_path}")


if __name__ == "__main__":
    run()

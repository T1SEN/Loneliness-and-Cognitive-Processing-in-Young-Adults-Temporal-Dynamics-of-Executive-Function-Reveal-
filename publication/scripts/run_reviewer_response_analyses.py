from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_participants, load_ucla_scores, load_dass_scores


OUTPUT_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "reviewer_response"


PRIMARY_DVS = [
    ("stroop_rt_interference_correct", "Stroop RT interference (correct)"),
    ("stroop_acc_interference", "Stroop accuracy interference"),
    ("wcst_categories_completed", "WCST categories completed"),
    ("wcst_perseverative_error_rate", "WCST perseverative error rate"),
    ("stroop_interference_slope", "Stroop interference RT slope"),
    ("stroop_rt_sd_incong", "Stroop RT SD (incongruent)"),
    ("wcst_post_shift_error_rt_mean", "WCST post-shift error RT mean"),
]

PRIMARY_ENDPOINTS = {
    "stroop_interference_slope",
    "wcst_post_shift_error_rt_mean",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def load_base_data() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    participants = load_participants(data_dir)
    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    dass = load_dass_scores(data_dir)

    base = participants.merge(ucla, on="participant_id", how="inner")
    base = base.merge(dass, on="participant_id", how="left")

    base["gender_male"] = base["gender"].map({"male": 1, "female": 0})
    base["dass_total"] = (
        base["dass_depression"].fillna(0)
        + base["dass_anxiety"].fillna(0)
        + base["dass_stress"].fillna(0)
    )
    return base


def _merge_features(base: pd.DataFrame, features: list[pd.DataFrame]) -> pd.DataFrame:
    merged = base.copy()
    for df in features:
        if df.empty:
            continue
        df = ensure_participant_id(df)
        cols = [c for c in df.columns if c != "participant_id"]
        overlap = [c for c in cols if c in merged.columns]
        if overlap:
            merged = merged.drop(columns=overlap)
        merged = merged.merge(df, on="participant_id", how="left")
    return merged


def load_task_data(task: str, base: pd.DataFrame) -> pd.DataFrame:
    task_dir = get_results_dir(task)
    features = []

    if task == "stroop":
        features.append(_read_csv(task_dir / "5_stroop_features.csv"))
        features.append(_read_csv(task_dir / "5_stroop_dynamic_drift_features.csv"))
    elif task == "wcst":
        features.append(_read_csv(task_dir / "5_wcst_features.csv"))
        features.append(_read_csv(task_dir / "5_wcst_switching_features.csv"))
    else:
        raise ValueError(f"Unsupported task: {task}")

    df = _merge_features(base, features)

    qc_ids = load_qc_ids(task)
    if qc_ids:
        df = df[df["participant_id"].isin(qc_ids)]

    return df


def load_qc_ids(task: str) -> set[str]:
    task_dir = get_results_dir(task)
    qc_ids_path = task_dir / "filtered_participant_ids.csv"
    if not qc_ids_path.exists():
        return set()
    qc_ids = _read_csv(qc_ids_path)
    qc_ids = ensure_participant_id(qc_ids)
    if "participant_id" not in qc_ids.columns:
        return set()
    return set(qc_ids["participant_id"].dropna().astype(str))


def add_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        std = df[col].std()
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std and np.isfinite(std) else np.nan
    return df


def fit_models(df: pd.DataFrame, dv: str, dass_total: bool = False) -> dict[str, float] | None:
    required = [dv, "z_ucla_score", "z_age", "gender_male"]
    if dass_total:
        required.append("z_dass_total")
    else:
        required += ["z_dass_depression", "z_dass_anxiety", "z_dass_stress"]
    sub = df[required].dropna()
    if len(sub) < 30:
        return None

    if dass_total:
        base_formula = f"{dv} ~ z_age + C(gender_male) + z_dass_total"
        full_formula = f"{dv} ~ z_age + C(gender_male) + z_dass_total + z_ucla_score"
    else:
        base_formula = (
            f"{dv} ~ z_age + C(gender_male) + z_dass_depression + "
            "z_dass_anxiety + z_dass_stress"
        )
        full_formula = (
            f"{dv} ~ z_age + C(gender_male) + z_dass_depression + "
            "z_dass_anxiety + z_dass_stress + z_ucla_score"
        )

    base_model = smf.ols(base_formula, data=sub).fit()
    full_model = smf.ols(full_formula, data=sub).fit()
    anova_res = anova_lm(base_model, full_model)
    delta_r2 = full_model.rsquared - base_model.rsquared
    p_ucla = float(anova_res["Pr(>F)"].iloc[1])

    return {
        "n": int(len(sub)),
        "ucla_beta": float(full_model.params.get("z_ucla_score", np.nan)),
        "ucla_se": float(full_model.bse.get("z_ucla_score", np.nan)),
        "ucla_p": float(full_model.pvalues.get("z_ucla_score", np.nan)),
        "delta_r2_ucla": float(delta_r2),
        "p_ucla_ols": p_ucla,
        "r2_full": float(full_model.rsquared),
    }


def fit_order_sensitivity(df: pd.DataFrame, dv: str) -> dict[str, float] | None:
    required = [
        dv,
        "z_ucla_score",
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
    ]
    sub = df[required].dropna()
    if len(sub) < 30:
        return None

    dass_formula = (
        f"{dv} ~ z_age + C(gender_male) + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress"
    )
    ucla_formula = f"{dv} ~ z_age + C(gender_male) + z_ucla_score"
    full_formula = (
        f"{dv} ~ z_age + C(gender_male) + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_ucla_score"
    )

    m_dass = smf.ols(dass_formula, data=sub).fit()
    m_ucla = smf.ols(ucla_formula, data=sub).fit()
    m_full = smf.ols(full_formula, data=sub).fit()

    anova_ucla = anova_lm(m_dass, m_full)
    anova_dass = anova_lm(m_ucla, m_full)

    return {
        "n": int(len(sub)),
        "delta_r2_ucla_given_dass": float(m_full.rsquared - m_dass.rsquared),
        "p_ucla_given_dass": float(anova_ucla["Pr(>F)"].iloc[1]),
        "delta_r2_dass_given_ucla": float(m_full.rsquared - m_ucla.rsquared),
        "p_dass_given_ucla": float(anova_dass["Pr(>F)"].iloc[1]),
    }


def add_fdr(df: pd.DataFrame, p_col: str = "ucla_p") -> pd.DataFrame:
    df = df.copy()
    pvals = df[p_col].to_numpy(dtype=float)
    mask = np.isfinite(pvals)
    df["ucla_fdr_q"] = np.nan
    df["ucla_fdr_sig"] = False
    if mask.any():
        _, qvals, _, _ = multipletests(pvals[mask], method="fdr_bh")
        df.loc[mask, "ucla_fdr_q"] = qvals
        df.loc[mask, "ucla_fdr_sig"] = qvals < 0.05
    return df


def compute_vif(df: pd.DataFrame) -> dict[str, float]:
    cols = ["z_ucla_score", "z_dass_depression", "z_dass_anxiety", "z_dass_stress", "z_age"]
    sub = df[cols].dropna()
    if len(sub) == 0:
        return {"vif_max": np.nan, "vif_mean": np.nan}
    X = sub.to_numpy()
    vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    return {"vif_max": float(np.max(vifs)), "vif_mean": float(np.mean(vifs))}


def winsorize_series(series: pd.Series, lower_q: float = 0.025, upper_q: float = 0.975) -> pd.Series:
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def compute_interference_slope(trials: pd.DataFrame, n_segments: int) -> pd.Series:
    records = []
    trials = trials.copy()
    trials = trials[trials["cond"].isin(["congruent", "incongruent"])]
    trials = trials[trials["correct"] == True]
    trials = trials[trials["is_rt_valid"] == True]

    for pid, grp in trials.groupby("participant_id"):
        grp = grp.sort_values("trial_index").reset_index(drop=True)
        n = len(grp)
        if n < max(20, n_segments * 5):
            records.append({"participant_id": pid, f"stroop_interference_slope_q{n_segments}": np.nan})
            continue

        order_rank = grp["trial_index"].rank(method="first")
        try:
            grp["segment"] = pd.qcut(order_rank, q=n_segments, labels=False, duplicates="drop")
        except ValueError:
            records.append({"participant_id": pid, f"stroop_interference_slope_q{n_segments}": np.nan})
            continue

        segment_vals = []
        for seg in sorted(grp["segment"].dropna().unique()):
            seg_df = grp[grp["segment"] == seg]
            inc = seg_df[seg_df["cond"] == "incongruent"]["rt_ms"]
            con = seg_df[seg_df["cond"] == "congruent"]["rt_ms"]
            if len(inc) >= 3 and len(con) >= 3:
                segment_vals.append((int(seg), float(inc.mean() - con.mean())))

        if len(segment_vals) < 3:
            slope = np.nan
        else:
            x = np.array([v[0] for v in segment_vals], dtype=float)
            y = np.array([v[1] for v in segment_vals], dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])

        records.append({"participant_id": pid, f"stroop_interference_slope_q{n_segments}": slope})

    return pd.DataFrame(records)


def split_half_reliability(trials: pd.DataFrame) -> dict[str, float]:
    def _compute_subset(df: pd.DataFrame) -> pd.DataFrame:
        return compute_interference_slope(df, n_segments=4)

    trials = trials.copy()
    trials = trials[trials["cond"].isin(["congruent", "incongruent"])]
    trials = trials[trials["correct"] == True]
    trials = trials[trials["is_rt_valid"] == True]
    trials["is_odd"] = trials["trial_index"] % 2 == 1

    odd = _compute_subset(trials[trials["is_odd"]])
    even = _compute_subset(trials[~trials["is_odd"]])

    merged = odd.merge(even, on="participant_id", how="inner", suffixes=("_odd", "_even"))
    if merged.empty:
        return {"n": 0, "r": np.nan, "spearman_brown": np.nan}

    x = merged["stroop_interference_slope_q4_odd"]
    y = merged["stroop_interference_slope_q4_even"]
    valid = x.notna() & y.notna()
    if valid.sum() < 5:
        return {"n": int(valid.sum()), "r": np.nan, "spearman_brown": np.nan}

    r = float(np.corrcoef(x[valid], y[valid])[0, 1])
    sb = (2 * r) / (1 + r) if np.isfinite(r) and r > -1 else np.nan
    return {"n": int(valid.sum()), "r": r, "spearman_brown": sb}


def bootstrap_ci_widths(trials: pd.DataFrame, n_boot: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    trials = trials.copy()
    trials = trials[trials["cond"].isin(["congruent", "incongruent"])]
    trials = trials[trials["correct"] == True]
    trials = trials[trials["is_rt_valid"] == True]

    for pid, grp in trials.groupby("participant_id"):
        grp = grp.sort_values("trial_index").reset_index(drop=True)
        if len(grp) < 30:
            continue
        slopes = []
        for _ in range(n_boot):
            sample = grp.sample(n=len(grp), replace=True, random_state=rng.integers(0, 1_000_000))
            slope_df = compute_interference_slope(sample, n_segments=4)
            slope_val = slope_df.iloc[0]["stroop_interference_slope_q4"]
            if pd.notna(slope_val):
                slopes.append(float(slope_val))
        if len(slopes) < 30:
            continue
        ci_low, ci_high = np.percentile(slopes, [2.5, 97.5])
        records.append({
            "participant_id": pid,
            "ci_width": float(ci_high - ci_low),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "n_boot": int(len(slopes)),
        })

    return pd.DataFrame(records)


def compute_wcst_shift_counts(trials: pd.DataFrame) -> pd.DataFrame:
    records = []
    trials = trials.copy()
    trials = trials.sort_values(["participant_id", "trial_index"])
    trials["cond"] = trials["cond"].astype(str).str.lower().replace({"color": "colour"})

    for pid, grp in trials.groupby("participant_id"):
        rules = grp["cond"].values
        correct = grp["correct"].astype(bool).values
        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        shift_trial_count = 0
        shift_error_count = 0

        for s_idx, idx in enumerate(change_indices):
            next_idx = change_indices[s_idx + 1] if s_idx + 1 < len(change_indices) else len(rules)
            reacq_idx = None
            for j in range(idx, next_idx - 2):
                if correct[j] and correct[j + 1] and correct[j + 2]:
                    reacq_idx = j + 2
                    break
            shift_end = reacq_idx if reacq_idx is not None else next_idx - 1
            shift_trial_count += max(0, shift_end - idx + 1)
            shift_error_count += int(np.sum(~correct[idx:shift_end + 1]))

        records.append({
            "participant_id": pid,
            "shift_trial_count": float(shift_trial_count),
            "post_shift_error_count": float(shift_error_count),
        })

    return pd.DataFrame(records)


def run_trial_level_ols(trials: pd.DataFrame, predictors: pd.DataFrame) -> dict[str, float] | None:
    df = trials.copy()
    df = df[df["is_rt_valid"] == True]
    df = df[df["timeout"] == False]
    df = df[df["correct"] == True]
    df = df[df["cond"].isin(["congruent", "incongruent", "neutral"])]

    if df.empty:
        return None

    df = df.sort_values(["participant_id", "trial_index"]).reset_index(drop=True)
    df["segment"] = np.nan

    for pid, grp in df.groupby("participant_id"):
        order_rank = grp["trial_index"].rank(method="first")
        try:
            seg = pd.qcut(order_rank, q=4, labels=False, duplicates="drop")
        except ValueError:
            continue
        df.loc[grp.index, "segment"] = seg.astype(float)

    df = df.dropna(subset=["segment"])
    df = df.merge(predictors, on="participant_id", how="inner")
    required = [
        "rt_ms",
        "segment",
        "z_ucla_score",
        "z_dass_depression",
        "z_dass_anxiety",
        "z_dass_stress",
        "z_age",
        "gender_male",
        "cond",
        "participant_id",
    ]
    df = df.dropna(subset=required)
    if len(df) < 1000:
        return None

    formula = (
        "rt_ms ~ segment * z_ucla_score + C(cond) + z_dass_depression + "
        "z_dass_anxiety + z_dass_stress + z_age + C(gender_male) + C(participant_id)"
    )
    model = smf.ols(formula, data=df).fit()
    term = "segment:z_ucla_score"
    return {
        "n_trials": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "beta": float(model.params.get(term, np.nan)),
        "se": float(model.bse.get(term, np.nan)),
        "t": float(model.tvalues.get(term, np.nan)),
        "p": float(model.pvalues.get(term, np.nan)),
    }


def compute_rt_variability(trials: pd.DataFrame, task: str) -> pd.DataFrame:
    df = trials.copy()
    if task == "stroop":
        df = df[df["cond"] == "incongruent"]
        df = df[df["correct"] == True]
        df = df[df["is_rt_valid"] == True]
    elif task == "wcst":
        df = df[df["is_rt_valid"] == True]
    else:
        raise ValueError(f"Unsupported task: {task}")

    records = []
    for pid, grp in df.groupby("participant_id"):
        rt = grp["rt_ms"].to_numpy(dtype=float)
        rt = rt[np.isfinite(rt)]
        if len(rt) < 10:
            continue
        records.append({
            "participant_id": pid,
            "rt_sd": float(np.std(rt, ddof=1)),
            "n_trials": int(len(rt)),
        })
    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_data()
    stroop = load_task_data("stroop", base)
    wcst = load_task_data("wcst", base)

    z_cols = ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age", "dass_total"]
    stroop = add_zscores(stroop, z_cols)
    wcst = add_zscores(wcst, z_cols)

    # Primary DVs
    rows = []
    for dv, label in PRIMARY_DVS:
        task_df = stroop if dv.startswith("stroop") else wcst
        res = fit_models(task_df, dv, dass_total=False)
        if res:
            res.update({
                "dv": dv,
                "label": label,
                "task": "stroop" if dv.startswith("stroop") else "wcst",
                "primary_endpoint": dv in PRIMARY_ENDPOINTS,
            })
            rows.append(res)

    primary_df = pd.DataFrame(rows)
    if not primary_df.empty:
        primary_df = add_fdr(primary_df, p_col="ucla_p")
        primary_df.to_csv(OUTPUT_DIR / "primary_dv_results.csv", index=False, encoding="utf-8-sig")

    # DASS total models
    rows = []
    for dv, label in PRIMARY_DVS:
        task_df = stroop if dv.startswith("stroop") else wcst
        res = fit_models(task_df, dv, dass_total=True)
        if res:
            res.update({
                "dv": dv,
                "label": label,
                "task": "stroop" if dv.startswith("stroop") else "wcst",
            })
            rows.append(res)
    dass_total_df = pd.DataFrame(rows)
    if not dass_total_df.empty:
        dass_total_df.to_csv(OUTPUT_DIR / "primary_dv_dass_total.csv", index=False, encoding="utf-8-sig")

    # Order sensitivity
    rows = []
    for dv, label in PRIMARY_DVS:
        task_df = stroop if dv.startswith("stroop") else wcst
        res = fit_order_sensitivity(task_df, dv)
        if res:
            res.update({
                "dv": dv,
                "label": label,
                "task": "stroop" if dv.startswith("stroop") else "wcst",
            })
            rows.append(res)
    order_df = pd.DataFrame(rows)
    if not order_df.empty:
        order_df.to_csv(OUTPUT_DIR / "primary_dv_order_sensitivity.csv", index=False, encoding="utf-8-sig")

    # VIF summary
    vif_rows = []
    for task, df in [("stroop", stroop), ("wcst", wcst)]:
        vif_vals = compute_vif(df)
        vif_vals["task"] = task
        vif_rows.append(vif_vals)
    vif_df = pd.DataFrame(vif_rows)
    vif_df.to_csv(OUTPUT_DIR / "vif_summary.csv", index=False, encoding="utf-8-sig")

    # Winsorized RT sensitivity
    winsor_rows = []
    rt_dvs = [dv for dv, _ in PRIMARY_DVS if "rt" in dv or "slope" in dv]
    for dv in rt_dvs:
        task_df = stroop if dv.startswith("stroop") else wcst
        if dv not in task_df.columns:
            continue
        tmp = task_df.copy()
        tmp[dv] = winsorize_series(tmp[dv])
        res = fit_models(tmp, dv, dass_total=False)
        if res:
            res.update({
                "dv": dv,
                "task": "stroop" if dv.startswith("stroop") else "wcst",
                "winsor": "2.5pct",
            })
            winsor_rows.append(res)
    winsor_df = pd.DataFrame(winsor_rows)
    if not winsor_df.empty:
        winsor_df.to_csv(OUTPUT_DIR / "winsorized_rt_results.csv", index=False, encoding="utf-8-sig")

    # Robust SE (HC3) for primary endpoints
    robust_rows = []
    for dv in PRIMARY_ENDPOINTS:
        task_df = stroop if dv.startswith("stroop") else wcst
        required = [
            dv,
            "z_ucla_score",
            "z_dass_depression",
            "z_dass_anxiety",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]
        sub = task_df[required].dropna()
        if len(sub) < 30:
            continue
        formula = (
            f"{dv} ~ z_ucla_score + z_dass_depression + z_dass_anxiety + "
            "z_dass_stress + z_age + C(gender_male)"
        )
        model = smf.ols(formula, data=sub).fit(cov_type="HC3")
        robust_rows.append({
            "dv": dv,
            "task": "stroop" if dv.startswith("stroop") else "wcst",
            "n": int(len(sub)),
            "ucla_beta": float(model.params.get("z_ucla_score", np.nan)),
            "ucla_se": float(model.bse.get("z_ucla_score", np.nan)),
            "ucla_p": float(model.pvalues.get("z_ucla_score", np.nan)),
        })
    robust_df = pd.DataFrame(robust_rows)
    if not robust_df.empty:
        robust_df.to_csv(OUTPUT_DIR / "primary_endpoints_hc3.csv", index=False, encoding="utf-8-sig")

    # Segment sensitivity (3, 4, 5, 6)
    stroop_trials_path = get_results_dir("stroop") / "4c_stroop_trials.csv"
    stroop_trials = _read_csv(stroop_trials_path)
    if not stroop_trials.empty:
        stroop_trials = ensure_participant_id(stroop_trials)
        qc_stroop_ids = load_qc_ids("stroop")
        if qc_stroop_ids:
            stroop_trials = stroop_trials[stroop_trials["participant_id"].isin(qc_stroop_ids)]
        seg_rows = []
        for n_segments in [3, 4, 5, 6]:
            seg_df = compute_interference_slope(stroop_trials, n_segments)
            merged = stroop.merge(seg_df, on="participant_id", how="left")
            dv = f"stroop_interference_slope_q{n_segments}"
            res = fit_models(merged, dv, dass_total=False)
            if res:
                res.update({
                    "dv": dv,
                    "task": "stroop",
                    "segments": n_segments,
                })
                seg_rows.append(res)
        seg_df = pd.DataFrame(seg_rows)
        if not seg_df.empty:
            seg_df.to_csv(OUTPUT_DIR / "stroop_segment_sensitivity.csv", index=False, encoding="utf-8-sig")

        # Split-half reliability
        split_res = split_half_reliability(stroop_trials)
        pd.DataFrame([split_res]).to_csv(
            OUTPUT_DIR / "stroop_split_half_reliability.csv",
            index=False,
            encoding="utf-8-sig",
        )

        # Bootstrap CI widths
        boot_df = bootstrap_ci_widths(stroop_trials, n_boot=100, seed=42)
        if not boot_df.empty:
            boot_df.to_csv(OUTPUT_DIR / "stroop_bootstrap_ci_widths.csv", index=False, encoding="utf-8-sig")
            summary = {
                "n_participants": int(len(boot_df)),
                "ci_width_mean": float(boot_df["ci_width"].mean()),
                "ci_width_sd": float(boot_df["ci_width"].std(ddof=1)),
                "ci_width_median": float(boot_df["ci_width"].median()),
                "ci_width_p25": float(boot_df["ci_width"].quantile(0.25)),
                "ci_width_p75": float(boot_df["ci_width"].quantile(0.75)),
            }
            pd.DataFrame([summary]).to_csv(
                OUTPUT_DIR / "stroop_bootstrap_ci_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

        # Trial-level OLS
        trial_level = run_trial_level_ols(
            stroop_trials,
            stroop[[
                "participant_id",
                "z_ucla_score",
                "z_dass_depression",
                "z_dass_anxiety",
                "z_dass_stress",
                "z_age",
                "gender_male",
            ]].dropna(),
        )
        if trial_level:
            pd.DataFrame([trial_level]).to_csv(
                OUTPUT_DIR / "stroop_trial_level_ols.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # WCST shift counts
    wcst_trials_path = get_results_dir("wcst") / "4b_wcst_trials.csv"
    wcst_trials = _read_csv(wcst_trials_path)
    if not wcst_trials.empty:
        wcst_trials = ensure_participant_id(wcst_trials)
        qc_wcst_ids = load_qc_ids("wcst")
        if qc_wcst_ids:
            wcst_trials = wcst_trials[wcst_trials["participant_id"].isin(qc_wcst_ids)]
        shift_df = compute_wcst_shift_counts(wcst_trials)
        if not shift_df.empty:
            shift_df.to_csv(OUTPUT_DIR / "wcst_shift_counts.csv", index=False, encoding="utf-8-sig")
            summary = {
                "n_participants": int(len(shift_df)),
                "shift_trial_mean": float(shift_df["shift_trial_count"].mean()),
                "shift_trial_sd": float(shift_df["shift_trial_count"].std(ddof=1)),
                "shift_trial_min": float(shift_df["shift_trial_count"].min()),
                "shift_trial_max": float(shift_df["shift_trial_count"].max()),
                "post_shift_error_mean": float(shift_df["post_shift_error_count"].mean()),
                "post_shift_error_sd": float(shift_df["post_shift_error_count"].std(ddof=1)),
                "post_shift_error_min": float(shift_df["post_shift_error_count"].min()),
                "post_shift_error_max": float(shift_df["post_shift_error_count"].max()),
            }
            pd.DataFrame([summary]).to_csv(
                OUTPUT_DIR / "wcst_shift_counts_summary.csv",
                index=False,
                encoding="utf-8-sig",
            )

    # RT variability summary and sensitivity
    var_rows = []
    sens_rows = []
    if not stroop_trials.empty:
        stroop_var = compute_rt_variability(stroop_trials, "stroop")
        if not stroop_var.empty:
            thresh = float(stroop_var["rt_sd"].quantile(0.975))
            var_rows.append({
                "task": "stroop",
                "n_participants": int(len(stroop_var)),
                "rt_sd_mean": float(stroop_var["rt_sd"].mean()),
                "rt_sd_sd": float(stroop_var["rt_sd"].std(ddof=1)),
                "rt_sd_min": float(stroop_var["rt_sd"].min()),
                "rt_sd_max": float(stroop_var["rt_sd"].max()),
                "rt_sd_p95": float(stroop_var["rt_sd"].quantile(0.95)),
                "rt_sd_p975": thresh,
            })
            keep_ids = set(stroop_var[stroop_var["rt_sd"] <= thresh]["participant_id"].astype(str))
            filtered = stroop[stroop["participant_id"].isin(keep_ids)]
            res = fit_models(filtered, "stroop_interference_slope", dass_total=False)
            if res:
                res.update({
                    "task": "stroop",
                    "dv": "stroop_interference_slope",
                    "filter": "rt_sd_p97_5",
                })
                sens_rows.append(res)

    if not wcst_trials.empty:
        wcst_var = compute_rt_variability(wcst_trials, "wcst")
        if not wcst_var.empty:
            thresh = float(wcst_var["rt_sd"].quantile(0.975))
            var_rows.append({
                "task": "wcst",
                "n_participants": int(len(wcst_var)),
                "rt_sd_mean": float(wcst_var["rt_sd"].mean()),
                "rt_sd_sd": float(wcst_var["rt_sd"].std(ddof=1)),
                "rt_sd_min": float(wcst_var["rt_sd"].min()),
                "rt_sd_max": float(wcst_var["rt_sd"].max()),
                "rt_sd_p95": float(wcst_var["rt_sd"].quantile(0.95)),
                "rt_sd_p975": thresh,
            })
            keep_ids = set(wcst_var[wcst_var["rt_sd"] <= thresh]["participant_id"].astype(str))
            filtered = wcst[wcst["participant_id"].isin(keep_ids)]
            res = fit_models(filtered, "wcst_post_shift_error_rt_mean", dass_total=False)
            if res:
                res.update({
                    "task": "wcst",
                    "dv": "wcst_post_shift_error_rt_mean",
                    "filter": "rt_sd_p97_5",
                })
                sens_rows.append(res)

    if var_rows:
        pd.DataFrame(var_rows).to_csv(
            OUTPUT_DIR / "rt_variability_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if sens_rows:
        pd.DataFrame(sens_rows).to_csv(
            OUTPUT_DIR / "rt_variability_sensitivity.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()

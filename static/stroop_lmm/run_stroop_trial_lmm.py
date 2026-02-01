from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "static").exists():
    ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from static.preprocessing.constants import OUTPUT_STATS_SUPP_DIR, get_results_dir
from static.preprocessing.core import ensure_participant_id
from static.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores


OUTPUT_DIR = OUTPUT_STATS_SUPP_DIR / "overall" / "stroop_lmm"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _coerce_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "t", "yes"})


def add_zscores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        std = df[col].std()
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std and np.isfinite(std) else np.nan
    return df


def load_base_data() -> pd.DataFrame:
    data_dir = get_results_dir("overall")
    participants = load_participants(data_dir)
    ucla = load_ucla_scores(data_dir).rename(columns={"ucla_total": "ucla_score"})
    dass = load_dass_scores(data_dir)

    base = participants.merge(ucla, on="participant_id", how="inner")
    base = base.merge(dass, on="participant_id", how="left")
    base["gender_male"] = base["gender"].map({"male": 1, "female": 0})
    return base


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


def prepare_stroop_trials() -> pd.DataFrame:
    trials_path = get_results_dir("overall") / "4a_stroop_trials.csv"
    trials = _read_csv(trials_path)
    if trials.empty:
        return trials

    trials = ensure_participant_id(trials)
    qc_ids = load_qc_ids("overall")
    if qc_ids:
        trials = trials[trials["participant_id"].isin(qc_ids)]

    trials["is_rt_valid"] = _coerce_bool(trials["is_rt_valid"])
    trials["timeout"] = _coerce_bool(trials["timeout"])
    trials["correct"] = _coerce_bool(trials["correct"])
    trials["cond"] = trials["cond"].astype(str).str.lower()
    trials["rt_ms"] = pd.to_numeric(trials["rt_ms"], errors="coerce")
    trials["trial_index"] = pd.to_numeric(trials["trial_index"], errors="coerce")

    trials = trials[trials["is_rt_valid"]]
    trials = trials[~trials["timeout"]]
    trials = trials[trials["correct"]]
    trials = trials[trials["cond"].isin({"congruent", "incongruent", "neutral"})]
    trials = trials.dropna(subset=["participant_id", "rt_ms", "trial_index"])
    if trials.empty:
        return trials

    trials = trials.sort_values(["participant_id", "trial_index"]).reset_index(drop=True)
    trials["segment"] = np.nan
    for pid, grp in trials.groupby("participant_id"):
        order_rank = grp["trial_index"].rank(method="first")
        try:
            seg = pd.qcut(order_rank, q=4, labels=False, duplicates="drop")
        except ValueError:
            continue
        trials.loc[grp.index, "segment"] = seg.astype(float)

    return trials.dropna(subset=["segment"])


def prepare_interference_trials() -> pd.DataFrame:
    trials_path = get_results_dir("overall") / "4a_stroop_trials.csv"
    trials = _read_csv(trials_path)
    if trials.empty:
        return trials

    trials = ensure_participant_id(trials)
    qc_ids = load_qc_ids("overall")
    if qc_ids:
        trials = trials[trials["participant_id"].isin(qc_ids)]

    trials["is_rt_valid"] = _coerce_bool(trials["is_rt_valid"])
    trials["timeout"] = _coerce_bool(trials["timeout"])
    trials["correct"] = _coerce_bool(trials["correct"])
    trials["cond"] = trials["cond"].astype(str).str.lower()
    trials["rt_ms"] = pd.to_numeric(trials["rt_ms"], errors="coerce")
    trials["trial_index"] = pd.to_numeric(trials["trial_index"], errors="coerce")

    trials = trials[trials["cond"].isin({"congruent", "incongruent"})]
    trials = trials[trials["is_rt_valid"]]
    trials = trials[~trials["timeout"]]
    trials = trials[trials["correct"]]
    trials = trials.dropna(subset=["participant_id", "rt_ms", "trial_index"])
    if trials.empty:
        return trials

    trials = trials.sort_values(["participant_id", "trial_index"]).reset_index(drop=True)
    # Continuous within-participant trial position (0-1 scale).
    def _scale_within(x: pd.Series) -> pd.Series:
        denom = x.max() - x.min()
        if denom == 0 or not np.isfinite(denom):
            return pd.Series([np.nan] * len(x), index=x.index)
        return (x - x.min()) / denom

    trials["trial_scaled"] = trials.groupby("participant_id")["trial_index"].transform(_scale_within)
    trials["cond_code"] = trials["cond"].map({"congruent": -0.5, "incongruent": 0.5})
    trials["log_rt"] = np.where(trials["rt_ms"] > 0, np.log(trials["rt_ms"]), np.nan)
    return trials.dropna(subset=["cond_code", "trial_scaled", "log_rt"])


def _fit_mixedlm(df: pd.DataFrame, re_formula: str, method: str) -> smf.mixedlm:
    formula = (
        "rt_ms ~ segment * z_ucla_score + C(cond) + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    model = smf.mixedlm(
        formula,
        data=df,
        groups=df["participant_id"],
        re_formula=re_formula,
    )
    return model.fit(reml=False, method=method, maxiter=200)


def _fit_mixedlm_with_warnings(
    formula: str,
    df: pd.DataFrame,
    re_formula: str,
    method: str,
) -> tuple[object, list[str]]:
    warning_msgs: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = smf.mixedlm(
            formula,
            data=df,
            groups=df["participant_id"],
            re_formula=re_formula,
        )
        result = model.fit(reml=False, method=method, maxiter=200)
    for warn in caught:
        if issubclass(warn.category, ConvergenceWarning):
            warning_msgs.append(str(warn.message))
    return result, warning_msgs


def fit_with_fallback(df: pd.DataFrame) -> tuple[object, dict[str, str], str | None]:
    attempts = [
        {"re_formula": "1 + segment", "method": "lbfgs"},
        {"re_formula": "1 + segment", "method": "powell"},
        {"re_formula": "1", "method": "lbfgs"},
        {"re_formula": "1", "method": "powell"},
    ]
    last_error = None
    last_result = None
    last_attempt = attempts[-1]

    for attempt in attempts:
        try:
            result = _fit_mixedlm(df, attempt["re_formula"], attempt["method"])
            last_result = result
            last_attempt = attempt
            if getattr(result, "converged", False):
                return result, attempt, None
        except Exception as exc:
            last_error = str(exc)

    if last_result is None:
        raise RuntimeError(f"MixedLM failed: {last_error}")
    return last_result, last_attempt, last_error or "not_converged"


def _get_term(result: object, term: str) -> dict[str, float]:
    params = getattr(result, "params", {})
    bse = getattr(result, "bse", {})
    tvalues = getattr(result, "tvalues", {})
    pvalues = getattr(result, "pvalues", {})
    return {
        "beta": float(params.get(term, np.nan)),
        "se": float(bse.get(term, np.nan)),
        "z": float(tvalues.get(term, np.nan)),
        "p": float(pvalues.get(term, np.nan)),
    }


def fit_interference_lmm(df: pd.DataFrame) -> tuple[object, dict[str, str], str | None]:
    formula = (
        "log_rt ~ trial_scaled * cond_code * z_ucla_score + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    attempts = [
        {"re_formula": "1 + trial_scaled + cond_code", "method": "lbfgs"},
        {"re_formula": "1 + trial_scaled", "method": "lbfgs"},
        {"re_formula": "1 + trial_scaled", "method": "powell"},
        {"re_formula": "1", "method": "lbfgs"},
        {"re_formula": "1", "method": "powell"},
    ]
    last_error = None
    last_result = None
    last_attempt = attempts[-1]

    for attempt in attempts:
        try:
            result, _ = _fit_mixedlm_with_warnings(
                formula,
                df,
                attempt["re_formula"],
                attempt["method"],
            )
            last_result = result
            last_attempt = attempt
            if getattr(result, "converged", False):
                return result, attempt, None
        except Exception as exc:
            last_error = str(exc)

    if last_result is None:
        raise RuntimeError(f"Interference MixedLM failed: {last_error}")
    return last_result, last_attempt, last_error or "not_converged"


def _summarize_interference_fit(
    result: object,
    meta: dict[str, object],
    warning_msgs: list[str],
) -> dict[str, object]:
    int_trial = _get_term(result, "trial_scaled")
    int_cond = _get_term(result, "cond_code")
    int_time = _get_term(result, "trial_scaled:cond_code")
    int_ucla = _get_term(result, "z_ucla_score")
    int_three = _get_term(result, "trial_scaled:cond_code:z_ucla_score")
    warning_text = " | ".join(warning_msgs)
    boundary_warning = any("boundary" in msg.lower() for msg in warning_msgs)

    return {
        **meta,
        "warning_count": len(warning_msgs),
        "boundary_warning": boundary_warning,
        "warning_msg": warning_text,
        "llf": float(getattr(result, "llf", np.nan)),
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "trial_scaled_beta": int_trial["beta"],
        "trial_scaled_se": int_trial["se"],
        "trial_scaled_z": int_trial["z"],
        "trial_scaled_p": int_trial["p"],
        "cond_beta": int_cond["beta"],
        "cond_se": int_cond["se"],
        "cond_z": int_cond["z"],
        "cond_p": int_cond["p"],
        "interference_time_beta": int_time["beta"],
        "interference_time_se": int_time["se"],
        "interference_time_z": int_time["z"],
        "interference_time_p": int_time["p"],
        "ucla_beta": int_ucla["beta"],
        "ucla_se": int_ucla["se"],
        "ucla_z": int_ucla["z"],
        "ucla_p": int_ucla["p"],
        "interference_time_x_ucla_beta": int_three["beta"],
        "interference_time_x_ucla_se": int_three["se"],
        "interference_time_x_ucla_z": int_three["z"],
        "interference_time_x_ucla_p": int_three["p"],
    }


def run_interference_variants(df: pd.DataFrame) -> pd.DataFrame:
    formula = (
        "log_rt ~ trial_scaled * cond_code * z_ucla_score + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    variants = [
        "1 + trial_scaled + cond_code",
        "1 + trial_scaled",
        "1",
    ]
    rows = []
    for re_formula in variants:
        fit_note = None
        warning_msgs: list[str] = []
        result = None
        method_used = "lbfgs"
        try:
            result, warning_msgs = _fit_mixedlm_with_warnings(
                formula,
                df,
                re_formula,
                method_used,
            )
        except Exception as exc:
            fit_note = str(exc)
            method_used = "powell"
            try:
                result, warning_msgs = _fit_mixedlm_with_warnings(
                    formula,
                    df,
                    re_formula,
                    method_used,
                )
            except Exception as exc2:
                fit_note = str(exc2)

        if result is None:
            rows.append({
                "n_trials": int(len(df)),
                "n_participants": int(df["participant_id"].nunique()),
                "re_formula": re_formula,
                "method": method_used,
                "converged": False,
                "fit_note": fit_note,
                "warning_count": len(warning_msgs),
                "boundary_warning": False,
                "warning_msg": " | ".join(warning_msgs),
            })
            continue

        meta = {
            "n_trials": int(len(df)),
            "n_participants": int(df["participant_id"].nunique()),
            "re_formula": re_formula,
            "method": method_used,
            "converged": bool(getattr(result, "converged", False)),
            "fit_note": fit_note,
        }
        rows.append(_summarize_interference_fit(result, meta, warning_msgs))

    return pd.DataFrame(rows)


def select_preferred_variant(variants: pd.DataFrame) -> pd.Series:
    if variants.empty:
        raise RuntimeError("No variant results available.")
    preference = [
        "1 + trial_scaled + cond_code",
        "1 + trial_scaled",
        "1",
    ]
    for re_formula in preference:
        subset = variants[
            (variants["re_formula"] == re_formula)
            & (variants["converged"] == True)
            & (variants["boundary_warning"] == False)
        ]
        if not subset.empty:
            return subset.iloc[0]
    for re_formula in preference:
        subset = variants[
            (variants["re_formula"] == re_formula)
            & (variants["converged"] == True)
        ]
        if not subset.empty:
            return subset.iloc[0]
    return variants.iloc[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_data()
    base = add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )

    trials = prepare_stroop_trials()
    if trials.empty:
        raise RuntimeError("No Stroop trials available after filtering.")

    predictors = base[
        [
            "participant_id",
            "z_ucla_score",
            "z_dass_depression",
            "z_dass_anxiety",
            "z_dass_stress",
            "z_age",
            "gender_male",
        ]
    ].dropna()
    predictors.to_csv(
        OUTPUT_DIR / "stroop_lmm_predictors.csv",
        index=False,
        encoding="utf-8-sig",
    )
    df = trials.merge(predictors, on="participant_id", how="inner")

    if df.empty:
        raise RuntimeError("No trials available after merging predictors.")

    result, attempt, fit_note = fit_with_fallback(df)

    interaction = _get_term(result, "segment:z_ucla_score")
    main_ucla = _get_term(result, "z_ucla_score")
    main_segment = _get_term(result, "segment")

    output = {
        "n_trials": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "re_formula": attempt["re_formula"],
        "method": attempt["method"],
        "converged": bool(getattr(result, "converged", False)),
        "fit_note": fit_note,
        "llf": float(getattr(result, "llf", np.nan)),
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "segment_beta": main_segment["beta"],
        "segment_se": main_segment["se"],
        "segment_z": main_segment["z"],
        "segment_p": main_segment["p"],
        "ucla_beta": main_ucla["beta"],
        "ucla_se": main_ucla["se"],
        "ucla_z": main_ucla["z"],
        "ucla_p": main_ucla["p"],
        "segment_x_ucla_beta": interaction["beta"],
        "segment_x_ucla_se": interaction["se"],
        "segment_x_ucla_z": interaction["z"],
        "segment_x_ucla_p": interaction["p"],
    }

    out_path = OUTPUT_DIR / "stroop_trial_level_lmm.csv"
    pd.DataFrame([output]).to_csv(out_path, index=False, encoding="utf-8-sig")

    # Interference slope mixed model (congruent vs incongruent only)
    int_trials = prepare_interference_trials()
    if int_trials.empty:
        raise RuntimeError("No interference trials available after filtering.")

    int_df = int_trials.merge(predictors, on="participant_id", how="inner")
    if int_df.empty:
        raise RuntimeError("No interference trials available after merging predictors.")

    variants = run_interference_variants(int_df)
    variants_path = OUTPUT_DIR / "stroop_interference_slope_lmm_variants.csv"
    variants.to_csv(variants_path, index=False, encoding="utf-8-sig")

    preferred = select_preferred_variant(variants)
    int_out_path = OUTPUT_DIR / "stroop_interference_slope_lmm.csv"
    preferred.to_frame().T.to_csv(int_out_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()


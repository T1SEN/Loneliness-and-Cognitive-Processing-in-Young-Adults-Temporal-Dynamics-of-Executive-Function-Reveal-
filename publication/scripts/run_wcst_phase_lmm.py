from __future__ import annotations

import sys
import warnings
from statistics import NormalDist
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from publication.preprocessing.constants import get_results_dir
from publication.preprocessing.core import ensure_participant_id
from publication.preprocessing.surveys import load_dass_scores, load_participants, load_ucla_scores
from publication.preprocessing.wcst._shared import prepare_wcst_trials


OUTPUT_DIR = ROOT / "publication" / "data" / "outputs" / "analysis" / "reviewer_response"
N_CATEGORIES = 6
PHASE_ORDER = ["exploitation", "exploration", "confirmation"]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


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


def _scale_within(x: pd.Series) -> pd.Series:
    denom = x.max() - x.min()
    if denom == 0 or not np.isfinite(denom):
        return pd.Series([np.nan] * len(x), index=x.index)
    return (x - x.min()) / denom


def label_wcst_phases(
    wcst: pd.DataFrame,
    rule_col: str,
    trial_col: str,
) -> pd.DataFrame:
    df = wcst.sort_values(["participant_id", trial_col]).copy()
    df["category_num"] = np.nan
    df["phase"] = pd.NA
    df["trial_in_category"] = np.nan
    df["trial_in_phase"] = np.nan
    df["in_shift_window"] = False

    for pid, grp in df.groupby("participant_id"):
        grp_sorted = grp.sort_values(trial_col).copy()
        idxs = grp_sorted.index.to_list()
        rules = (
            grp_sorted[rule_col].astype(str).str.lower().replace({"color": "colour"}).to_numpy()
        )
        correct = grp_sorted["correct"].astype(bool).to_numpy()

        change_indices = [i for i in range(1, len(rules)) if rules[i] != rules[i - 1]]
        segment_starts = [0] + change_indices
        segment_ends = change_indices + [len(rules)]

        for cat_idx, (start, end) in enumerate(zip(segment_starts, segment_ends), start=1):
            if cat_idx > N_CATEGORIES:
                break
            if start >= end:
                continue

            reacq_start = None
            reacq_idx = None
            for j in range(start, end - 2):
                if correct[j] and correct[j + 1] and correct[j + 2]:
                    reacq_start = j
                    reacq_idx = j + 2
                    break
            shift_end_err = reacq_idx if reacq_idx is not None else end - 1

            for i in range(start, end):
                row_idx = idxs[i]
                df.at[row_idx, "category_num"] = float(cat_idx)
                df.at[row_idx, "trial_in_category"] = float(i - start + 1)
                df.at[row_idx, "in_shift_window"] = bool(i <= shift_end_err)

                if reacq_start is None:
                    df.at[row_idx, "phase"] = "exploration"
                    df.at[row_idx, "trial_in_phase"] = float(i - start + 1)
                elif i < reacq_start:
                    df.at[row_idx, "phase"] = "exploration"
                    df.at[row_idx, "trial_in_phase"] = float(i - start + 1)
                elif i <= reacq_idx:
                    df.at[row_idx, "phase"] = "confirmation"
                    df.at[row_idx, "trial_in_phase"] = float(i - reacq_start + 1)
                else:
                    df.at[row_idx, "phase"] = "exploitation"
                    df.at[row_idx, "trial_in_phase"] = float(i - reacq_idx)

    df["phase"] = pd.Categorical(df["phase"], categories=PHASE_ORDER)
    df["phase_code"] = df["phase"].map({"exploitation": 0, "exploration": 1, "confirmation": 2})
    df["progress_in_phase"] = df.groupby(
        ["participant_id", "category_num", "phase"],
        dropna=False,
        observed=False,
    )["trial_in_phase"].transform(_scale_within)
    return df


def add_post_error_flags(df: pd.DataFrame, trial_col: str) -> pd.DataFrame:
    df = df.sort_values(["participant_id", trial_col]).copy()
    df["prev_correct"] = df.groupby("participant_id")["correct"].shift(1)
    df["post_error"] = df["prev_correct"] == False
    df["post_error"] = df["post_error"].astype("boolean").fillna(False)
    df["post_error_code"] = df["post_error"].astype(int)
    return df


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


def _summarize_terms(result: object) -> dict[str, float]:
    terms = {
        "phase_exploration": _get_term(
            result, "C(phase, Treatment('exploitation'))[T.exploration]"
        ),
        "phase_confirmation": _get_term(
            result, "C(phase, Treatment('exploitation'))[T.confirmation]"
        ),
        "ucla_main": _get_term(result, "z_ucla_score"),
        "exploration_x_ucla": _get_term(
            result, "C(phase, Treatment('exploitation'))[T.exploration]:z_ucla_score"
        ),
        "confirmation_x_ucla": _get_term(
            result, "C(phase, Treatment('exploitation'))[T.confirmation]:z_ucla_score"
        ),
        "post_error": _get_term(result, "post_error_code"),
        "post_error_x_ucla": _get_term(result, "post_error_code:z_ucla_score"),
        "category_num": _get_term(result, "category_num"),
    }

    flat: dict[str, float] = {}
    for key, stats in terms.items():
        for stat_key, value in stats.items():
            flat[f"{key}_{stat_key}"] = value
    return flat


def run_model_variants(
    df: pd.DataFrame,
    formula: str,
    model_label: str,
    re_formulas: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for re_formula in re_formulas:
        method_used = "lbfgs"
        fit_note = None
        warning_msgs: list[str] = []
        result = None
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

        warning_text = " | ".join(warning_msgs)
        boundary_warning = any("boundary" in msg.lower() for msg in warning_msgs)
        row = {
            "model": model_label,
            "n_trials": int(len(df)),
            "n_participants": int(df["participant_id"].nunique()),
            "re_formula": re_formula,
            "method": method_used,
            "converged": bool(getattr(result, "converged", False)) if result is not None else False,
            "fit_note": fit_note,
            "warning_count": len(warning_msgs),
            "boundary_warning": boundary_warning,
            "warning_msg": warning_text,
            "llf": float(getattr(result, "llf", np.nan)) if result is not None else np.nan,
            "aic": float(getattr(result, "aic", np.nan)) if result is not None else np.nan,
            "bic": float(getattr(result, "bic", np.nan)) if result is not None else np.nan,
        }
        if result is not None:
            row.update(_summarize_terms(result))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_model_with_fallback(
    df: pd.DataFrame,
    formula: str,
    re_formulas: list[str],
) -> tuple[object | None, dict[str, object]]:
    attempts: list[dict[str, object]] = []
    result = None
    for re_formula in re_formulas:
        for method in ("lbfgs", "powell"):
            warning_msgs: list[str] = []
            fit_note = None
            try:
                result, warning_msgs = _fit_mixedlm_with_warnings(
                    formula,
                    df,
                    re_formula,
                    method,
                )
            except Exception as exc:
                fit_note = str(exc)
                result = None
            attempts.append({
                "re_formula": re_formula,
                "method": method,
                "converged": bool(getattr(result, "converged", False)) if result is not None else False,
                "warning_msgs": warning_msgs,
                "fit_note": fit_note,
            })
            if result is not None and getattr(result, "converged", False) and not warning_msgs:
                return result, attempts[-1]
    return result, attempts[-1] if attempts else {"re_formula": None, "method": None}


def linear_combo(
    result: object,
    terms: list[str],
) -> dict[str, float]:
    params = getattr(result, "params", {})
    if not isinstance(params, pd.Series):
        params = pd.Series(params)
    cov = getattr(result, "cov_params", lambda: None)()
    coef = 0.0
    for term in terms:
        coef += float(params.get(term, 0.0))
    if cov is None:
        return {"beta": coef, "se": np.nan, "z": np.nan, "p": np.nan}
    if isinstance(cov, pd.DataFrame):
        cov_df = cov.copy()
    else:
        cov_df = pd.DataFrame(cov, index=params.index, columns=params.index)
    vec = np.zeros(len(cov_df))
    term_index = list(cov_df.index)
    for term in terms:
        if term in term_index:
            vec[term_index.index(term)] += 1.0
    var = float(vec.T @ cov_df.to_numpy() @ vec)
    se = np.sqrt(var) if var >= 0 else np.nan
    z = coef / se if se and np.isfinite(se) else np.nan
    p = 2 * (1 - NormalDist().cdf(abs(z))) if np.isfinite(z) else np.nan
    return {"beta": coef, "se": se, "z": z, "p": p}


def run_phase_specific_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    formula = (
        "log_rt ~ z_ucla_score + category_num + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    for phase in PHASE_ORDER:
        subset = df[df["phase"] == phase].copy()
        if subset.empty:
            continue
        result, meta = fit_model_with_fallback(subset, formula, ["1"])
        if result is None:
            rows.append({
                "phase": phase,
                "n_trials": int(len(subset)),
                "n_participants": int(subset["participant_id"].nunique()),
                "re_formula": meta.get("re_formula"),
                "method": meta.get("method"),
                "converged": False,
                "ucla_beta": np.nan,
                "ucla_se": np.nan,
                "ucla_z": np.nan,
                "ucla_p": np.nan,
                "fit_note": meta.get("fit_note"),
            })
            continue
        ucla = _get_term(result, "z_ucla_score")
        rows.append({
            "phase": phase,
            "n_trials": int(len(subset)),
            "n_participants": int(subset["participant_id"].nunique()),
            "re_formula": meta.get("re_formula"),
            "method": meta.get("method"),
            "converged": bool(getattr(result, "converged", False)),
            "ucla_beta": ucla["beta"],
            "ucla_se": ucla["se"],
            "ucla_z": ucla["z"],
            "ucla_p": ucla["p"],
            "fit_note": meta.get("fit_note"),
        })
    return pd.DataFrame(rows)


def run_category_phase_models(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    df_num = df.copy()
    df_num["category_num"] = df_num["category_num"].astype(float)
    df_cat = df.copy()
    df_cat["category_num"] = df_cat["category_num"].astype(int)
    df_cat["category_num"] = pd.Categorical(
        df_cat["category_num"],
        categories=list(range(1, N_CATEGORIES + 1)),
        ordered=True,
    )
    phase_term = "C(phase, Treatment('exploitation'))"
    cat_term = "C(category_num, Treatment(1))"
    base_covariates = (
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )

    formula_linear = (
        f"log_rt ~ category_num * {phase_term} * z_ucla_score + "
        "post_error_code * z_ucla_score + "
        f"{base_covariates}"
    )
    linear_result, linear_meta = fit_model_with_fallback(df_num, formula_linear, ["1"])
    if linear_result is not None:
        linear_terms = {
            "category_x_phase_x_ucla_exploration": _get_term(
                linear_result, "category_num:C(phase, Treatment('exploitation'))[T.exploration]:z_ucla_score"
            ),
            "category_x_phase_x_ucla_confirmation": _get_term(
                linear_result, "category_num:C(phase, Treatment('exploitation'))[T.confirmation]:z_ucla_score"
            ),
            "category_x_ucla_exploitation": _get_term(
                linear_result, "category_num:z_ucla_score"
            ),
        }
        linear_rows = []
        for label, stats in linear_terms.items():
            linear_rows.append({
                "term": label,
                "beta": stats["beta"],
                "se": stats["se"],
                "z": stats["z"],
                "p": stats["p"],
                "re_formula": linear_meta.get("re_formula"),
                "method": linear_meta.get("method"),
                "converged": bool(getattr(linear_result, "converged", False)),
                "fit_note": linear_meta.get("fit_note"),
            })
        results["linear"] = pd.DataFrame(linear_rows)

    formula_cat = (
        f"log_rt ~ {cat_term} * {phase_term} * z_ucla_score + "
        "post_error_code * z_ucla_score + "
        f"{base_covariates}"
    )
    cat_result, cat_meta = fit_model_with_fallback(df_cat, formula_cat, ["1"])
    if cat_result is not None:
        params = getattr(cat_result, "params", pd.Series(dtype=float))
        bse = getattr(cat_result, "bse", pd.Series(dtype=float))
        tvals = getattr(cat_result, "tvalues", pd.Series(dtype=float))
        pvals = getattr(cat_result, "pvalues", pd.Series(dtype=float))
        terms_df = pd.DataFrame({
            "term": params.index,
            "beta": params.values,
            "se": bse.reindex(params.index).values,
            "z": tvals.reindex(params.index).values,
            "p": pvals.reindex(params.index).values,
        })
        terms_df["re_formula"] = cat_meta.get("re_formula")
        terms_df["method"] = cat_meta.get("method")
        terms_df["converged"] = bool(getattr(cat_result, "converged", False))
        terms_df["fit_note"] = cat_meta.get("fit_note")
        results["categorical_terms"] = terms_df

        slope_rows: list[dict[str, object]] = []
        for category in range(1, N_CATEGORIES + 1):
            for phase in PHASE_ORDER:
                terms = ["z_ucla_score"]
                if category != 1:
                    terms.append(f"{cat_term}[T.{category}]:z_ucla_score")
                if phase != "exploitation":
                    terms.append(f"{phase_term}[T.{phase}]:z_ucla_score")
                if category != 1 and phase != "exploitation":
                    terms.append(
                        f"{cat_term}[T.{category}]:{phase_term}[T.{phase}]:z_ucla_score"
                    )
                stats = linear_combo(cat_result, terms)
                slope_rows.append({
                    "category_num": category,
                    "phase": phase,
                    "ucla_beta": stats["beta"],
                    "ucla_se": stats["se"],
                    "ucla_z": stats["z"],
                    "ucla_p": stats["p"],
                })
        results["categorical_slopes"] = pd.DataFrame(slope_rows)

    return results


def select_preferred_variant(variants: pd.DataFrame, model_label: str) -> pd.Series:
    subset = variants[variants["model"] == model_label]
    preference = ["1 + phase_code", "1"]
    for re_formula in preference:
        cand = subset[
            (subset["re_formula"] == re_formula)
            & (subset["converged"] == True)
            & (subset["boundary_warning"] == False)
        ]
        if not cand.empty:
            return cand.iloc[0]
    for re_formula in preference:
        cand = subset[
            (subset["re_formula"] == re_formula)
            & (subset["converged"] == True)
        ]
        if not cand.empty:
            return cand.iloc[0]
    return subset.iloc[0]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_data()
    base = add_zscores(
        base,
        ["ucla_score", "dass_depression", "dass_anxiety", "dass_stress", "age"],
    )

    prepared = prepare_wcst_trials()
    wcst = prepared["wcst"]
    rt_col = prepared["rt_col"]
    trial_col = prepared["trial_col"]
    rule_col = prepared["rule_col"]
    if not isinstance(wcst, pd.DataFrame) or wcst.empty or rt_col is None or trial_col is None or rule_col is None:
        raise RuntimeError("WCST trials missing required columns.")

    qc_ids = load_qc_ids("wcst")
    if qc_ids:
        wcst = wcst[wcst["participant_id"].isin(qc_ids)]

    wcst["rt_ms"] = pd.to_numeric(wcst[rt_col], errors="coerce")
    wcst = wcst[wcst["rt_ms"].notna()]
    if "is_rt_valid" in wcst.columns:
        wcst = wcst[wcst["is_rt_valid"] == True]

    wcst["rule"] = wcst[rule_col].astype(str).str.lower().replace({"color": "colour"})
    wcst = label_wcst_phases(wcst, rule_col="rule", trial_col=trial_col)
    wcst = add_post_error_flags(wcst, trial_col=trial_col)
    wcst["log_rt"] = np.where(wcst["rt_ms"] > 0, np.log(wcst["rt_ms"]), np.nan)

    wcst = wcst.dropna(subset=["phase", "category_num", "log_rt", "phase_code"])

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

    df = wcst.merge(predictors, on="participant_id", how="inner")
    df = df.dropna(subset=["log_rt"])

    if df.empty:
        raise RuntimeError("No WCST trials available after merging predictors.")

    phase_counts = (
        df.groupby(["phase"], observed=False)
        .size()
        .reset_index(name="n_trials")
        .sort_values("phase")
    )
    phase_counts.to_csv(
        OUTPUT_DIR / "wcst_phase_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )

    phase_term = "C(phase, Treatment('exploitation'))"
    formula_base = (
        f"log_rt ~ {phase_term} * z_ucla_score + "
        "post_error_code * z_ucla_score + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )
    formula_ext = (
        f"log_rt ~ {phase_term} * z_ucla_score + "
        "post_error_code * z_ucla_score + category_num + C(rule) + "
        "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male"
    )

    re_formulas = ["1 + phase_code", "1"]
    variants = []
    variants.append(run_model_variants(df, formula_base, "base", re_formulas))
    variants.append(run_model_variants(df, formula_ext, "extended", re_formulas))
    variants_df = pd.concat(variants, ignore_index=True)
    variants_df.to_csv(
        OUTPUT_DIR / "wcst_phase_lmm_variants.csv",
        index=False,
        encoding="utf-8-sig",
    )

    base_pref = select_preferred_variant(variants_df, "base")
    ext_pref = select_preferred_variant(variants_df, "extended")
    base_pref.to_frame().T.to_csv(
        OUTPUT_DIR / "wcst_phase_lmm_base.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ext_pref.to_frame().T.to_csv(
        OUTPUT_DIR / "wcst_phase_lmm_extended.csv",
        index=False,
        encoding="utf-8-sig",
    )

    phase_specific = run_phase_specific_models(df)
    phase_specific.to_csv(
        OUTPUT_DIR / "wcst_phase_lmm_absolute_ucla.csv",
        index=False,
        encoding="utf-8-sig",
    )

    category_results = run_category_phase_models(df)
    if "linear" in category_results:
        category_results["linear"].to_csv(
            OUTPUT_DIR / "wcst_category_phase_ucla_linear.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if "categorical_terms" in category_results:
        category_results["categorical_terms"].to_csv(
            OUTPUT_DIR / "wcst_category_phase_ucla_categorical_terms.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if "categorical_slopes" in category_results:
        category_results["categorical_slopes"].to_csv(
            OUTPUT_DIR / "wcst_category_phase_ucla_categorical_slopes.csv",
            index=False,
            encoding="utf-8-sig",
        )


if __name__ == "__main__":
    main()

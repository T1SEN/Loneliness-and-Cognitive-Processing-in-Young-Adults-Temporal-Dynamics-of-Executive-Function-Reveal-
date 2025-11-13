#!/usr/bin/env Rscript
#
# Loneliness × Executive Function analysis pipeline
# -------------------------------------------------
# This script ingests the exported CSV files produced by the Flutter tasks
# (participants info, survey scores, and cognitive summaries), constructs
# task-level and latent executive-function indicators, and runs the core
# frequentist tests requested in the IRB plan:
#   1) Does trait loneliness (UCLA v3) predict each EF readout after
#      controlling for DASS-21 Depression/Anxiety/Stress?
#   2) Does loneliness load onto a shared “meta-control” factor spanning
#      Stroop, PRP, and WCST metrics?
# The script prints model summaries to the console and saves tidy p-value
# tables to `results/analysis_outputs/`.
#
# Usage (from repo root):
#   Rscript analysis/loneliness_exec_models.R
# -------------------------------------------------

suppressPackageStartupMessages({
  pkgs <- c("tidyverse", "janitor", "broom", "psych", "glue")
  need <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
  if (length(need) > 0) install.packages(need)
  lapply(pkgs, library, character.only = TRUE)
})

data_dir <- "results"
output_dir <- file.path(data_dir, "analysis_outputs")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

message("1) Loading CSV exports …")
participants <- read_csv(file.path(data_dir, "1_participants_info.csv"),
                         show_col_types = FALSE) |>
  clean_names()

surveys <- read_csv(file.path(data_dir, "2_surveys_results.csv"),
                    show_col_types = FALSE) |>
  clean_names()

cog <- read_csv(file.path(data_dir, "3_cognitive_tests_summary.csv"),
                show_col_types = FALSE) |>
  clean_names()

message(glue("   • Participants: {nrow(participants)} rows"))
message(glue("   • Survey rows : {nrow(surveys)} rows"))
message(glue("   • Cognitive   : {nrow(cog)} rows"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
zscore <- function(x) {
  if (all(is.na(x))) return(rep(NA_real_, length(x)))
  if (sd(x, na.rm = TRUE) == 0) return(rep(0, length(x)))
  as.numeric(scale(x))
}

clean_gender <- function(x) {
  case_when(
    is.na(x) ~ NA_character_,
    str_detect(x, "\uB0A8") ~ "male",   # Korean char U+B0A8 ("nam")
    str_detect(x, "\uC5EC") ~ "female", # Korean char U+C5EC ("yeo")
    TRUE ~ NA_character_
  )
}

# ---------------------------------------------------------------------------
# Surveys: derive UCLA trait loneliness + DASS state mood controls
# ---------------------------------------------------------------------------
ucla <- surveys |>
  filter(str_to_lower(survey_name) == "ucla") |>
  transmute(participant_id, ucla_total = score)

dass <- surveys |>
  filter(str_to_lower(survey_name) == "dass") |>
  transmute(
    participant_id,
    dass_anx = score_a,
    dass_stress = score_s,
    dass_dep = score_d
  ) |>
  mutate(dass_total = dass_anx + dass_stress + dass_dep)

# ---------------------------------------------------------------------------
# Cognitive summaries → task-specific EF indicators
# ---------------------------------------------------------------------------
prp <- cog |>
  filter(str_to_lower(test_name) == "prp") |>
  transmute(
    participant_id,
    prp_n_trials = n_trials,
    prp_acc_t1 = acc_t1,
    prp_acc_t2 = acc_t2,
    prp_mrt_t1 = mrt_t1,
    prp_mrt_t2 = mrt_t2,
    prp_rt_short = rt2_soa_50,
    prp_rt_long = rt2_soa_1200,
    prp_bottleneck = rt2_soa_50 - rt2_soa_1200,
    prp_rt_slope = (rt2_soa_50 - rt2_soa_1200) / 1150
  )

stroop <- cog |>
  filter(str_to_lower(test_name) == "stroop") |>
  transmute(
    participant_id,
    stroop_total_trials = total,
    stroop_accuracy = accuracy,
    stroop_effect = stroop_effect,
    stroop_mrt_incong = mrt_incong,
    stroop_mrt_cong = mrt_cong,
    stroop_mrt_total = mrt_total
  )

wcst <- cog |>
  filter(str_to_lower(test_name) == "wcst") |>
  transmute(
    participant_id,
    wcst_total_errors = total_error_count,
    wcst_persev_errors = perseverative_error_count,
    wcst_nonpersev_errors = non_perseverative_error_count,
    wcst_completed_categories = completed_categories,
    wcst_conceptual_pct = conceptual_level_responses_percent,
    wcst_persev_resp_pct = perseverative_responses_percent,
    wcst_failure_to_maintain_set = failure_to_maintain_set
  )

# ---------------------------------------------------------------------------
# Merge everything
# ---------------------------------------------------------------------------
analysis_df <- participants |>
  select(participant_id, age, gender) |>
  mutate(
    age = as.numeric(age),
    gender = factor(clean_gender(gender))
  ) |>
  left_join(ucla, by = "participant_id") |>
  left_join(dass, by = "participant_id") |>
  left_join(prp, by = "participant_id") |>
  left_join(stroop, by = "participant_id") |>
  left_join(wcst, by = "participant_id")

message(glue("2) Combined analytic dataset: {nrow(analysis_df)} participants"))

# Z-scores for focal predictors
analysis_df <- analysis_df |>
  mutate(
    z_ucla = zscore(ucla_total),
    z_dass_dep = zscore(dass_dep),
    z_dass_anx = zscore(dass_anx),
    z_dass_stress = zscore(dass_stress)
  )

# ---------------------------------------------------------------------------
# Latent meta-control factor (1-factor PCA on EF metrics)
# ---------------------------------------------------------------------------
ef_vars <- c("stroop_effect", "prp_bottleneck", "wcst_total_errors")
ef_complete <- analysis_df |>
  select(participant_id, all_of(ef_vars)) |>
  drop_na()

if (nrow(ef_complete) >= 15) {
  message("3) Estimating 1-factor PCA (meta-control) …")
  pca_fit <- psych::principal(scale(ef_complete |> select(-participant_id)),
                              nfactors = 1, rotate = "none", scores = TRUE)
  meta_scores <- tibble(
    participant_id = ef_complete$participant_id,
    meta_control = as.numeric(pca_fit$scores[, 1])
  )
  analysis_df <- analysis_df |>
    left_join(meta_scores, by = "participant_id")
  write_csv(
    tibble(
      indicator = ef_vars,
      loading = as.numeric(pca_fit$loadings[, 1])
    ),
    file.path(output_dir, "meta_control_loadings.csv")
  )
  message("   • PCA loadings saved to meta_control_loadings.csv")
} else {
  message("3) PCA skipped (need ≥15 complete cases).")
  analysis_df$meta_control <- NA_real_
}

# ---------------------------------------------------------------------------
# Frequentist models: Loneliness predicting EF outcomes
# ---------------------------------------------------------------------------
model_specs <- tribble(
  ~outcome, ~nice_name,
  "stroop_effect", "Stroop interference (ms)",
  "prp_bottleneck", "PRP bottleneck (short-long RT)",
  "wcst_total_errors", "WCST total errors",
  "meta_control", "Latent meta-control factor"
)

fit_lm <- function(dep, data) {
  data_mod <- data |>
    select(
      !!sym(dep), z_ucla, z_dass_dep, z_dass_anx, z_dass_stress,
      age, gender
    ) |>
    rename(y = !!sym(dep)) |>
    drop_na(y, z_ucla, z_dass_dep, z_dass_anx, z_dass_stress)
  if (nrow(data_mod) < 20) return(NULL)
  lm(y ~ z_ucla + z_dass_dep + z_dass_anx + z_dass_stress +
       age + gender,
     data = data_mod)
}

message("4) Running hierarchical linear models …")
model_tbl <- model_specs |>
  mutate(
    model = map(outcome, ~fit_lm(.x, analysis_df)),
    valid = map_lgl(model, ~!is.null(.x))
  ) |>
  filter(valid)

if (nrow(model_tbl) == 0) {
  stop("No models could be estimated (insufficient complete cases).")
}

coef_tbl <- model_tbl |>
  mutate(
    coefs = map2(model, nice_name, ~tidy(.x, conf.int = TRUE) |>
                   mutate(outcome = .y))
  ) |>
  select(coefs) |>
  unnest(coefs) |>
  relocate(outcome, term)

fit_tbl <- model_tbl |>
  mutate(
    glance = map2(model, nice_name, ~glance(.x) |> mutate(outcome = .y))
  ) |>
  select(glance) |>
  unnest(glance) |>
  relocate(outcome, r.squared, adj.r.squared, p.value, AIC, BIC, nobs)

write_csv(coef_tbl, file.path(output_dir, "loneliness_models_coefficients.csv"))
write_csv(fit_tbl, file.path(output_dir, "loneliness_models_fit.csv"))

message("   • Coefficients saved to loneliness_models_coefficients.csv")
message("   • Model fit stats saved to loneliness_models_fit.csv")

cat("\n=== Key p-values for UCLA Loneliness ===\n")
coef_tbl |>
  filter(term == "z_ucla") |>
  arrange(p.value) |>
  mutate(across(c(estimate, std.error, conf.low, conf.high, p.value),
                ~round(., 4))) |>
  print(n = Inf)

message("\nDone.")

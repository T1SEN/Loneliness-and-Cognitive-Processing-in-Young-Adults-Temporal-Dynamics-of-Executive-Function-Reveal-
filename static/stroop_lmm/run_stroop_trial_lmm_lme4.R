#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("lme4", quietly = TRUE)) {
    stop("lme4 is required but not installed.")
  }
  library(lme4)
  has_lmerTest <- requireNamespace("lmerTest", quietly = TRUE)
  if (has_lmerTest) {
    library(lmerTest)
  }
})

args <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
script_path <- if (length(file_arg) > 0) sub("^--file=", "", file_arg[1]) else ""
root <- if (nzchar(script_path)) dirname(dirname(script_path)) else getwd()
if (!dir.exists(file.path(root, "static"))) {
  root <- if (nzchar(script_path)) dirname(dirname(dirname(script_path))) else getwd()
}

output_dir <- file.path(root, "outputs", "stats", "analysis", "overall", "stroop_lmm")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

trials_path <- file.path(root, "data", "complete_overall", "4a_stroop_trials.csv")
trials <- read.csv(trials_path, fileEncoding = "UTF-8-BOM")

if (!"participant_id" %in% names(trials) && "participantId" %in% names(trials)) {
  trials$participant_id <- trials$participantId
}

to_bool <- function(x) {
  if (is.logical(x)) return(x)
  lx <- tolower(as.character(x))
  lx %in% c("true", "1", "t", "yes")
}

trials$is_rt_valid <- to_bool(trials$is_rt_valid)
trials$timeout <- to_bool(trials$timeout)
trials$correct <- to_bool(trials$correct)
trials$cond <- tolower(as.character(trials$cond))
trials$rt_ms <- suppressWarnings(as.numeric(trials$rt_ms))
trials$trial_index <- suppressWarnings(as.numeric(trials$trial_index))

trials <- trials[trials$cond %in% c("congruent", "incongruent"), ]
trials <- trials[trials$is_rt_valid, ]
trials <- trials[!trials$timeout, ]
trials <- trials[trials$correct, ]
trials <- trials[!is.na(trials$participant_id) & !is.na(trials$rt_ms) & !is.na(trials$trial_index), ]

scale_within <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (length(rng) < 2 || diff(rng) == 0) {
    return(rep(NA_real_, length(x)))
  }
  (x - rng[1]) / diff(rng)
}

trials$trial_scaled <- ave(trials$trial_index, trials$participant_id, FUN = scale_within)
trials$cond_code <- ifelse(trials$cond == "incongruent", 0.5, -0.5)
trials$log_rt <- ifelse(trials$rt_ms > 0, log(trials$rt_ms), NA_real_)
trials <- trials[!is.na(trials$trial_scaled) & !is.na(trials$cond_code) & !is.na(trials$log_rt), ]

pred_path <- file.path(output_dir, "stroop_lmm_predictors.csv")
if (!file.exists(pred_path)) {
  stop("Missing predictors file: ", pred_path)
}
pred <- read.csv(pred_path, fileEncoding = "UTF-8-BOM")

df <- merge(trials, pred, by = "participant_id", all = FALSE)
if (nrow(df) == 0) {
  stop("No data after merging predictors.")
}

fit_model <- function(re_formula, optimizer = "bobyqa") {
  warn_msgs <- character()
  formula <- as.formula(
    paste(
      "log_rt ~ trial_scaled * cond_code * z_ucla_score +",
      "z_dass_depression + z_dass_anxiety + z_dass_stress + z_age + gender_male +",
      sprintf("(%s | participant_id)", re_formula)
    )
  )
  fit <- withCallingHandlers(
    lmer(formula, data = df, REML = FALSE, control = lmerControl(optimizer = optimizer)),
    warning = function(w) {
      warn_msgs <<- c(warn_msgs, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  summ <- summary(fit)
  coefs <- summ$coefficients
  term <- "trial_scaled:cond_code:z_ucla_score"
  if (!term %in% rownames(coefs)) {
    stop("Missing term in model: ", term)
  }
  p_col <- if ("Pr(>|t|)" %in% colnames(coefs)) "Pr(>|t|)" else NA
  p_val <- if (is.na(p_col)) NA_real_ else coefs[term, p_col]
  p_method <- if (is.na(p_col)) "none" else "lmerTest"

  data.frame(
    n_trials = nrow(df),
    n_participants = length(unique(df$participant_id)),
    re_formula = re_formula,
    optimizer = optimizer,
    singular = isSingular(fit, tol = 1e-4),
    warning_count = length(warn_msgs),
    warning_msg = paste(warn_msgs, collapse = " | "),
    beta = coefs[term, "Estimate"],
    se = coefs[term, "Std. Error"],
    t = coefs[term, "t value"],
    p = p_val,
    p_method = p_method,
    stringsAsFactors = FALSE
  )
}

variants <- c("1 + trial_scaled + cond_code", "1 + trial_scaled", "1")
rows <- lapply(variants, function(re_formula) fit_model(re_formula))
out <- do.call(rbind, rows)

out_path <- file.path(output_dir, "stroop_interference_slope_lme4.csv")
write.csv(out, out_path, row.names = FALSE)


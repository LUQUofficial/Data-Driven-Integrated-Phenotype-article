# Input: merged_data, with one row per participant.
library(xgboost)
library(dplyr)
library(openxlsx)

# Features, domains and outcomes
features <- c(
  "Light_RA", "Light_IS", "Light_IV", "Light_M10", "Light_L5",
  "Light_M10_start", "Light_L5_start", "NightLux_gt10_avg",
  "Mesor", "RA", "IS", "IV", "M10", "L5", "M10_start", "L5_start",
  "Active_midpoint", "Rest_midpoint", "Rest_midpoint_SD",
  "sleep_duration", "sleep_midpoint", "Sleep_midpoint_SD",
  "Sleep_pct", "Sedentary_pct", "LPA_pct", "MVPA_pct"
)
domain_features <- setdiff(features, c("NightLux_gt10_avg", "Mesor"))
feature_group <- setNames(
  c(rep("Light", 7), rep("RAR", 10), rep("Sleep", 3), rep("Movement", 4)),
  domain_features
)
outcomes <- c("whobmiz", "WHtR_z", "WHR_z")
params <- list(objective = "reg:squarederror", max_depth = 3,
               eta = 0.05, subsample = 0.8, colsample_bytree = 0.8,
               nthread = 1)
nrounds <- 300
B <- 1000
output_dir <- "results/xgboost_shap"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
performance <- feature_importance <- domain_importance <- bootstrap_results <- list()
models <- shap_values <- list()

for (outcome in outcomes) {
  # Outcome-specific complete-case sample
  dat <- merged_data %>%
    select(all_of(c(features, outcome))) %>%
    filter(if_all(everything(), ~ !is.na(.x)))
  X <- as.matrix(dat[, features])
  y <- dat[[outcome]]
  n <- nrow(X)
  stopifnot(is.numeric(X), is.numeric(y), all(is.finite(X)), all(is.finite(y)))

  # Proposed evaluation: 80% training and 20% held-out testing
  set.seed(123)
  train_id <- sample(seq_len(n), floor(0.8 * n))
  test_id <- setdiff(seq_len(n), train_id)
  stopifnot(length(test_id) > 1, stats::sd(y) > 0)
  dtrain <- xgb.DMatrix(X[train_id, , drop = FALSE], label = y[train_id])
  dtest <- xgb.DMatrix(X[test_id, , drop = FALSE])
  set.seed(123)
  train_model <- xgb.train(params, dtrain, nrounds = nrounds, verbose = 0)
  pred <- predict(train_model, dtest)
  sst <- sum((y[test_id] - mean(y[test_id]))^2)
  performance[[outcome]] <- data.frame(
    Outcome = outcome, N = n, N_train = length(train_id), N_test = length(test_id),
    RMSE = sqrt(mean((y[test_id] - pred)^2)),
    MAE = mean(abs(y[test_id] - pred)),
    R2 = if (sst > 0) 1 - sum((y[test_id] - pred)^2) / sst else NA_real_
  )

  # Refit on the full analysis sample and calculate SHAP values
  dall <- xgb.DMatrix(X, label = y)
  set.seed(123)
  model <- xgb.train(params, dall, nrounds = nrounds, verbose = 0)
  shap <- predict(model, dall, predcontrib = TRUE)[, seq_along(features), drop = FALSE]
  colnames(shap) <- features  # Exclude the final BIAS column.
  importance <- colMeans(abs(shap))
  feature_table <- data.frame(
    Outcome = outcome, Feature = features, Domain = unname(feature_group[features]),
    Mean_abs_SHAP = importance, row.names = NULL
  )
  feature_table$Percent <- 100 * importance / sum(importance)
  feature_table$Rank <- rank(-importance, ties.method = "min")
  domain_importance[[outcome]] <- feature_table %>%
    filter(!is.na(Domain)) %>%
    group_by(Outcome, Domain) %>%
    summarise(Mean_abs_SHAP = sum(Mean_abs_SHAP), .groups = "drop") %>%
    mutate(Percent = 100 * Mean_abs_SHAP / sum(Mean_abs_SHAP))
  models[[outcome]] <- model
  shap_values[[outcome]] <- shap

  # Bootstrap stability: refit, then explain the SAME original sample
  boot_shap <- boot_rank <- matrix(NA_real_, B, length(features))
  for (b in seq_len(B)) {
    set.seed(1000 + b)
    idx <- sample(seq_len(n), n, replace = TRUE)
    dboot <- xgb.DMatrix(X[idx, , drop = FALSE], label = y[idx])
    # Fixed model seed; variation comes from resampling the participants.
    set.seed(123)
    boot_model <- xgb.train(params, dboot, nrounds = nrounds, verbose = 0)
    boot_values <- predict(boot_model, dall, predcontrib = TRUE)[,
                        seq_along(features), drop = FALSE]
    boot_shap[b, ] <- colMeans(abs(boot_values))
    boot_rank[b, ] <- rank(-boot_shap[b, ], ties.method = "min")
    if (b %% 100 == 0) message(outcome, ": bootstrap ", b, "/", B)
  }
  feature_table$Bootstrap_SHAP_Q025 <- apply(boot_shap, 2, quantile, probs = 0.025)
  feature_table$Bootstrap_SHAP_Q975 <- apply(boot_shap, 2, quantile, probs = 0.975)
  feature_table$Bootstrap_rank_median <- apply(boot_rank, 2, median)
  feature_table$Bootstrap_rank_Q025 <- apply(boot_rank, 2, quantile, probs = 0.025)
  feature_table$Bootstrap_rank_Q975 <- apply(boot_rank, 2, quantile, probs = 0.975)
  feature_importance[[outcome]] <- arrange(feature_table, Rank)
  bootstrap_results[[outcome]] <- data.frame(
    Outcome = outcome, Iteration = rep(seq_len(B), each = length(features)),
    Feature = rep(features, times = B),
    Mean_abs_SHAP = as.vector(t(boot_shap)), Rank = as.vector(t(boot_rank))
  )
}

# Input: merged_data, with one row per participant and a unique Subject ID.
library(dplyr)
library(cluster)
library(factoextra)
library(ggplot2)
library(openxlsx)

# 24 behavioral metrics
#Light M10, light L5,  M10 and L5 were transformed using ln(1+x) before analysis. 
metrics <- c(
  "Light_RA", "Light_IS", "Light_IV", "Light_M10", "Light_L5",
  "Light_M10_start", "Light_L5_start",
  "RA", "IS", "IV", "M10", "L5", "M10_start", "L5_start",
  "Active_midpoint", "Rest_midpoint", "Rest_midpoint_SD",
  "sleep_duration", "sleep_midpoint", "Sleep_midpoint_SD",
  "Sleep_pct", "Sedentary_pct", "LPA_pct", "MVPA_pct"
)
pca_data <- merged_data %>%
  select(Subject, all_of(metrics)) %>%
  filter(if_all(all_of(metrics), ~ !is.na(.x)))
stopifnot(!anyNA(merged_data$Subject), !anyDuplicated(merged_data$Subject))
x <- as.matrix(pca_data[, metrics])
stopifnot(is.numeric(x), all(is.finite(x)), all(apply(x, 2, sd) > 0))
cat("Complete-case sample size:", nrow(pca_data), "\n")
# Pearson correlation matrix
pearson_r <- cor(x, method = "pearson")
# Standardization and PCA
pca_scaled <- scale(x)
pca_result <- prcomp(pca_scaled, center = FALSE, scale. = FALSE)
variance <- pca_result$sdev^2 / sum(pca_result$sdev^2)

PCA_table <- data.frame(
  PC = paste0("PC", seq_along(variance)),
  Variance_percent = 100 * variance,
  Cumulative_percent = 100 * cumsum(variance)
)
# Retain components explaining at least 80% of cumulative variance
n_pc <- which(cumsum(variance) >= 0.80)[1]
pc_for_cluster <- pca_result$x[, seq_len(n_pc), drop = FALSE]
# Elbow method
set.seed(123)
elbow_plot <- fviz_nbclust(
  pc_for_cluster, kmeans, method = "wss",
  k.max = 10, nstart = 100
) + ggtitle("Elbow method")
# Silhouette method
set.seed(123)
silhouette_plot <- fviz_nbclust(
  pc_for_cluster, kmeans, method = "silhouette",
  k.max = 10, nstart = 100
) + ggtitle("Silhouette method")
# Davies–Bouldin index
library(clusterSim)
DB_table <- data.frame(k = 2:10, DB_index = NA_real_)
set.seed(123)
for (i in seq_len(nrow(DB_table))) {
  km <- kmeans(pc_for_cluster, centers = DB_table$k[i], nstart = 100)
  DB_table$DB_index[i] <- index.DB(
    pc_for_cluster, km$cluster,
    centrotypes = "centroids", p = 2, q = 1
  )$DB
}


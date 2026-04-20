# ============================================================
#  Smart Energy — R Statistical Analysis
#  Compatible with dataset.txt (space/tab separated, 1-2 cols)
# ============================================================

library(ggplot2)
library(gridExtra)
library(scales)
library(dplyr)

# ── 1. Load Data ─────────────────────────────────────────────
data_path <- "/app/data/dataset.txt"
csv_path  <- "/app/data/hourly_data.csv"

if (file.exists(csv_path)) {
  cat("Loading from hourly_data.csv\n")
  data <- read.csv(csv_path, stringsAsFactors = FALSE)
  # Standardise column name
  if (!"Global_active_power" %in% colnames(data)) {
    numeric_cols <- sapply(data, is.numeric)
    colnames(data)[which(numeric_cols)[1]] <- "Global_active_power"
  }
} else if (file.exists(data_path)) {
  cat("Loading from dataset.txt\n")
  raw <- read.table(data_path, header = FALSE, sep = "",
                    fill = TRUE, stringsAsFactors = FALSE,
                    comment.char = "", quote = "")
  # Pick the first numeric column as power reading
  numeric_cols <- sapply(raw, function(x) suppressWarnings(!all(is.na(as.numeric(x)))))
  power_col <- which(numeric_cols)[1]
  data <- data.frame(
    Global_active_power = suppressWarnings(as.numeric(raw[[power_col]])),
    row_index = seq_len(nrow(raw))
  )
  data <- data[!is.na(data$Global_active_power), ]
  data <- data[data$Global_active_power > 0 & data$Global_active_power <= 20, ]
} else {
  cat("No data file found — generating synthetic data\n")
  set.seed(42)
  n <- 8760
  data <- data.frame(
    Global_active_power = abs(rnorm(n, mean = 5, sd = 1.5)),
    row_index = 1:n
  )
}

cat(sprintf("Rows loaded: %d\n", nrow(data)))
cat(sprintf("Power range: %.3f — %.3f kW\n",
            min(data$Global_active_power, na.rm = TRUE),
            max(data$Global_active_power, na.rm = TRUE)))

power <- data$Global_active_power
n     <- length(power)

# ── 2. Simple anomaly flag (z-score > 3) ─────────────────────
mu    <- mean(power, na.rm = TRUE)
sigma <- sd(power,   na.rm = TRUE)
data$anomaly <- abs((power - mu) / sigma) > 3
data$idx     <- seq_len(nrow(data))

n_anomalies <- sum(data$anomaly, na.rm = TRUE)
cat(sprintf("Anomalies detected: %d (%.2f%%)\n", n_anomalies, 100 * n_anomalies / n))

# ── 3. Dark theme ─────────────────────────────────────────────
dark_theme <- theme(
  plot.background  = element_rect(fill = "#0d1117", colour = NA),
  panel.background = element_rect(fill = "#161b22", colour = NA),
  panel.grid.major = element_line(colour = "#30363d"),
  panel.grid.minor = element_line(colour = "#21262d"),
  text             = element_text(colour = "#e6edf3"),
  axis.text        = element_text(colour = "#8b949e"),
  axis.title       = element_text(colour = "#c9d1d9"),
  plot.title       = element_text(colour = "#58a6ff", size = 11, face = "bold"),
  legend.background = element_rect(fill = "#161b22"),
  legend.text      = element_text(colour = "#e6edf3")
)

# ── 4. Plot 1: Histogram ──────────────────────────────────────
p1 <- ggplot(data, aes(x = Global_active_power)) +
  geom_histogram(bins = 60, fill = "#58a6ff", colour = "#1f6feb", alpha = 0.8) +
  labs(title = "Power Consumption Distribution",
       x = "Power (kW)", y = "Frequency") +
  dark_theme

# ── 5. Plot 2: Time Series with Anomalies ────────────────────
sample_size <- min(5000, nrow(data))
sample_data <- data[seq(1, nrow(data), length.out = sample_size), ]

p2 <- ggplot(sample_data, aes(x = idx, y = Global_active_power)) +
  geom_line(colour = "#58a6ff", alpha = 0.6, linewidth = 0.3) +
  geom_point(data = sample_data[sample_data$anomaly == TRUE, ],
             aes(x = idx, y = Global_active_power),
             colour = "#f85149", size = 1.5) +
  labs(title = "Time Series with Anomalies (red)",
       x = "Sample Index", y = "Power (kW)") +
  dark_theme

# ── 6. Plot 3: Boxplot ────────────────────────────────────────
p3 <- ggplot(data, aes(y = Global_active_power)) +
  geom_boxplot(fill = "#1f6feb", colour = "#58a6ff", alpha = 0.7,
               outlier.colour = "#f85149", outlier.size = 0.8) +
  labs(title = "Power Quartile Boxplot", y = "Power (kW)", x = "") +
  dark_theme

# ── 7. Plot 4: Rolling Mean ───────────────────────────────────
window <- max(1, floor(n / 200))
data$rolling_mean <- stats::filter(power, rep(1/window, window), sides = 2)

p4 <- ggplot(data[seq(1, nrow(data), length.out = sample_size), ],
             aes(x = idx)) +
  geom_line(aes(y = Global_active_power), colour = "#8b949e", alpha = 0.4, linewidth = 0.2) +
  geom_line(aes(y = rolling_mean), colour = "#3fb950", linewidth = 0.8, na.rm = TRUE) +
  labs(title = "Rolling Mean Overlay (green)",
       x = "Index", y = "Power (kW)") +
  dark_theme

# ── 8. Plot 5: Kernel Density ─────────────────────────────────
p5 <- ggplot(data, aes(x = Global_active_power)) +
  geom_density(fill = "#388bfd", colour = "#58a6ff", alpha = 0.5) +
  labs(title = "Kernel Density Estimate",
       x = "Power (kW)", y = "Density") +
  dark_theme

# ── 9. Plot 6: Summary Stats Table ───────────────────────────
stats_df <- data.frame(
  Metric = c("N", "Mean", "Median", "Std Dev", "Min", "Max",
             "Q1", "Q3", "Anomalies", "Anomaly %"),
  Value  = c(
    formatC(n, format = "d", big.mark = ","),
    sprintf("%.3f kW", mean(power, na.rm = TRUE)),
    sprintf("%.3f kW", median(power, na.rm = TRUE)),
    sprintf("%.3f kW", sd(power, na.rm = TRUE)),
    sprintf("%.3f kW", min(power, na.rm = TRUE)),
    sprintf("%.3f kW", max(power, na.rm = TRUE)),
    sprintf("%.3f kW", quantile(power, 0.25, na.rm = TRUE)),
    sprintf("%.3f kW", quantile(power, 0.75, na.rm = TRUE)),
    formatC(n_anomalies, format = "d"),
    sprintf("%.2f%%", 100 * n_anomalies / n)
  )
)

p6 <- ggplot(stats_df, aes(x = 0, y = rev(seq_along(Metric)))) +
  geom_text(aes(label = paste0(Metric, ":  ", Value)),
            hjust = 0, colour = "#e6edf3", size = 3.2, x = 0.05) +
  xlim(0, 1) +
  labs(title = "Summary Statistics") +
  theme_void() +
  theme(
    plot.background = element_rect(fill = "#161b22", colour = NA),
    plot.title = element_text(colour = "#58a6ff", size = 11,
                              face = "bold", hjust = 0.05, vjust = 1)
  )

# ── 10. Save 6-panel plot ─────────────────────────────────────
out_path <- "/app/data/plot.png"
png(out_path, width = 1400, height = 900, res = 120, bg = "#0d1117")
grid.arrange(p1, p2, p3, p4, p5, p6, nrow = 2)
dev.off()

cat(sprintf("Plot saved to %s\n", out_path))
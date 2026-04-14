data <- read.csv("processed_spark_data.csv")

data$Global_active_power <- as.numeric(as.character(data$Global_active_power))

# Remove NA values
data <- na.omit(data)

png("plot.png")

hist(
  data$Global_active_power,
  col="skyblue",
  main="Power Consumption Distribution",
  xlab="Power",
  border="white"
)

dev.off()
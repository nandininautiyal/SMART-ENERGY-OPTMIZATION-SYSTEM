from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("EnergyProcessing").getOrCreate()


df = spark.read.csv(
    "data/dataset.txt",
    header=True,
    inferSchema=True,
    sep=";"
)

print("Columns:", df.columns)

# Now columns will work
df = df.select("Global_active_power")

pdf = df.toPandas()

pdf.to_csv("data/processed_spark_data.csv", index=False)

print("Spark processing complete!")
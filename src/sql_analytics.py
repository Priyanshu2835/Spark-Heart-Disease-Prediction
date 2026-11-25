
import glob
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, count, avg, round as spark_round
)

spark = (
    SparkSession.builder
    .appName("INT315_SparkSQL_Analytics")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

files = glob.glob("data/heart_disease_clean.csv/*.csv")

if not files:
    raise FileNotFoundError(
        "Cleaned CSV not found. Run src/preprocessing.py first."
    )

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(files[0])
)

df.createOrReplaceTempView("heart_disease")

print("\n===== SPARK SQL ANALYTICS =====")

print("\n1. Dataset summary:")
df.select(
    count("*").alias("total_records"),
    spark_round(avg("age"), 2).alias("average_age"),
    spark_round(avg("chol"), 2).alias("average_cholesterol"),
    spark_round(avg("trestbps"), 2).alias("average_blood_pressure")
).show(truncate=False)

print("\n2. Diagnosis distribution using SQL:")
spark.sql("""
    SELECT label, COUNT(*) AS total_records
    FROM heart_disease
    GROUP BY label
    ORDER BY label
""").show()

print("\n3. Age group analysis:")
age_groups = df.withColumn(
    "age_group",
    when(col("age") < 40, "Under 40")
    .when(col("age") < 50, "40-49")
    .when(col("age") < 60, "50-59")
    .otherwise("60 and above")
)

age_groups.createOrReplaceTempView("age_analysis")

spark.sql("""
    SELECT
        age_group,
        COUNT(*) AS total_records,
        SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)
            AS diagnosis_positive,
        ROUND(AVG(chol), 2) AS average_cholesterol
    FROM age_analysis
    GROUP BY age_group
    ORDER BY MIN(age)
""").show(truncate=False)

print("\n4. Average measurements by diagnosis label:")
spark.sql("""
    SELECT
        label,
        COUNT(*) AS total_records,
        ROUND(AVG(age), 2) AS average_age,
        ROUND(AVG(chol), 2) AS average_cholesterol,
        ROUND(AVG(trestbps), 2) AS average_resting_bp,
        ROUND(AVG(thalach), 2) AS average_max_heart_rate
    FROM heart_disease
    GROUP BY label
    ORDER BY label
""").show(truncate=False)

print("\n5. Top 10 records by cholesterol:")
df.select("age", "sex", "chol", "trestbps", "label") \
  .orderBy(col("chol").desc()) \
  .show(10, truncate=False)

print("\n===== SPARK SQL ANALYTICS COMPLETED =====")

spark.stop()

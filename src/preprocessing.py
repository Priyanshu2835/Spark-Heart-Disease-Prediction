
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.sql.types import StructType, StructField, DoubleType

spark = (
    SparkSession.builder
    .appName("INT315_Healthcare_Preprocessing")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num"
]

schema = StructType([
    StructField(name, DoubleType(), True)
    for name in columns
])

df = (
    spark.read
    .option("header", "false")
    .option("nullValue", "?")
    .schema(schema)
    .csv("data/processed.cleveland.data")
)

print("\n===== PREPROCESSING STARTED =====")
original_count = df.count()
print("Original records:", original_count)

# Impute missing feature values using each column's median.
# The target column 'num' is not imputed.
feature_columns = [c for c in columns if c != "num"]

for name in feature_columns:
    median = df.approxQuantile(name, [0.5], 0.0)[0]
    df = df.fillna({name: median})

# Create a binary classification label:
# 0 = num is 0; 1 = num is greater than 0.
df = df.withColumn(
    "label",
    when(col("num") == 0, 0.0).otherwise(1.0)
)

# Verify that no feature values or targets remain missing.
print("\n===== MISSING VALUES AFTER CLEANING =====")
df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
]).show(truncate=False)

print("\n===== BINARY CLASS DISTRIBUTION =====")
df.groupBy("label").count().orderBy("label").show()

print("\n===== CLEANED DATA PREVIEW =====")
df.show(10, truncate=False)

print("\n===== SAVING CLEANED DATASET =====")
output_path = "data/heart_disease_clean.csv"

(
    df.coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(output_path)
)

print("Saved cleaned dataset to:", output_path)
print("Final records:", df.count())
print("Final columns:", len(df.columns))

spark.stop()

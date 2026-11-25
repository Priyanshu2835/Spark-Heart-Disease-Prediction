from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import col, count, when

spark = (
    SparkSession.builder
    .appName("INT315_DatasetInspection")
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
    StructField(name, DoubleType(), True) for name in columns
])

df = (
    spark.read
    .option("header", "false")
    .option("nullValue", "?")
    .schema(schema)
    .csv("data/processed.cleveland.data")
)

print("\n===== DATASET OVERVIEW =====")
print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))
print("Column names:", df.columns)

print("\n===== DATA TYPES =====")
df.printSchema()

print("\n===== FIRST 10 RECORDS =====")
df.show(10, truncate=False)

print("\n===== MISSING VALUES =====")
missing = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in columns
])
missing.show(truncate=False)

print("\n===== ORIGINAL DIAGNOSIS DISTRIBUTION =====")
df.groupBy("num").count().orderBy("num").show()

spark.stop()
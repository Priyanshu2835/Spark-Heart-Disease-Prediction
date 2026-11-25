
import csv
import glob
import os

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

spark = (
    SparkSession.builder
    .appName("INT315_Correlation_Analysis")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

files = glob.glob("data/heart_disease_clean.csv/*.csv")
if not files:
    raise FileNotFoundError("Run preprocessing.py first.")

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(files[0])
)

# Include numerical health measurements and the binary target.
columns = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "label"
]

assembler = VectorAssembler(
    inputCols=columns,
    outputCol="correlation_features"
)

vector_df = assembler.transform(df).select("correlation_features")

correlation_matrix = Correlation.corr(
    vector_df,
    "correlation_features",
    "pearson"
).first()[0].toArray()

print("\n===== PEARSON CORRELATION ANALYSIS =====")
print("Records analysed:", df.count())
print("Method: Pearson correlation coefficient")
print("Range: -1 (negative) to +1 (positive)")
print("Label: 0 = no disease; 1 = disease present")

print("\nCorrelation matrix:")
print(f"{'Variable':<12}", end="")
for name in columns:
    print(f"{name:>12}", end="")
print()

for i, name in enumerate(columns):
    print(f"{name:<12}", end="")
    for j in range(len(columns)):
        print(f"{correlation_matrix[i][j]:>12.3f}", end="")
    print()

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/correlation_matrix.csv"

with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Variable"] + columns)

    for i, name in enumerate(columns):
        writer.writerow(
            [name] + [
                round(float(correlation_matrix[i][j]), 4)
                for j in range(len(columns))
            ]
        )

print("\nCorrelation matrix saved to:", output_path)

spark.stop()
print("\n===== CORRELATION ANALYSIS COMPLETED =====")

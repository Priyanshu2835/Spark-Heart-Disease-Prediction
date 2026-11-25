
import csv
import glob
import os
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count

spark = (
    SparkSession.builder
    .appName("INT315_Spark_Benchmark")
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
).cache()

total_records = df.count()

print("\n===== SPARK EXECUTION BENCHMARK =====")
print("Dataset records:", total_records)
print("Execution mode: local[*]")
print("Operation: group by label and calculate averages")
print("Note: timings include partitioning and execution overhead.")

# Warm up Spark so initial startup costs are less likely
# to dominate the measured runs.
df.groupBy("label").count().collect()

results = []

for partitions in [1, 2, 4]:
    partitioned_df = df.repartition(partitions)

    start = time.perf_counter()

    summary = (
        partitioned_df
        .groupBy("label")
        .agg(
            count("*").alias("record_count"),
            avg("age").alias("average_age"),
            avg("chol").alias("average_cholesterol"),
            avg("thalach").alias("average_max_heart_rate")
        )
        .collect()
    )

    elapsed = time.perf_counter() - start

    result = {
        "Partitions": partitions,
        "Elapsed Seconds": round(elapsed, 6),
        "Records": total_records
    }
    results.append(result)

    print(f"\nPartitions: {partitions}")
    print(f"Elapsed seconds: {elapsed:.6f}")
    print("Aggregation results:")

    for row in sorted(summary, key=lambda r: r["label"]):
        print(
            f"Label {int(row['label'])}: "
            f"count={row['record_count']}, "
            f"avg_age={row['average_age']:.2f}, "
            f"avg_chol={row['average_cholesterol']:.2f}, "
            f"avg_thalach={row['average_max_heart_rate']:.2f}"
        )

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/spark_benchmark.csv"

with open(output_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["Partitions", "Elapsed Seconds", "Records"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\n===== BENCHMARK SUMMARY =====")
print(f"{'Partitions':<15}{'Elapsed Seconds':>20}{'Records':>12}")

for result in results:
    print(
        f"{result['Partitions']:<15}"
        f"{result['Elapsed Seconds']:>20.6f}"
        f"{result['Records']:>12}"
    )

print("\nBenchmark saved to:", output_path)

df.unpersist()
spark.stop()

print("\n===== BENCHMARK COMPLETED =====")

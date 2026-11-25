
import csv
import glob
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("INT315_RDD_Analytics")
    .master("local[*]")
    .getOrCreate()
)

sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Locate the CSV part file written by Spark.
files = glob.glob("data/heart_disease_clean.csv/*.csv")

if not files:
    raise FileNotFoundError(
        "Cleaned CSV not found. Run src/preprocessing.py first."
    )

csv_path = files[0]

# Read CSV rows using Spark Core RDDs.
lines = sc.textFile(csv_path)
header = lines.first()

data_lines = lines.filter(lambda line: line != header)

# Parse CSV records.
rows = data_lines.map(lambda line: next(csv.reader([line])))

# Convert rows into (age, label) pairs.
age_label = rows.map(
    lambda r: (float(r[0]), int(float(r[14])))
)

print("\n===== SPARK CORE / RDD ANALYTICS =====")

print("\n1. Total records using RDD count():")
print(data_lines.count())

print("\n2. Sample records using map():")
print(age_label.take(5))

print("\n3. Patients aged 60 or above using filter():")
older_patients = age_label.filter(lambda x: x[0] >= 60)
print("Count:", older_patients.count())

print("\n4. Count by binary diagnosis label using reduceByKey():")
class_counts = (
    age_label
    .map(lambda x: (x[1], 1))
    .reduceByKey(lambda a, b: a + b)
    .sortByKey()
    .collect()
)

for label, count in class_counts:
    print("Label", label, ":", count, "records")

print("\n5. Total age sum using reduce():")
total_age = age_label.map(lambda x: x[0]).reduce(
    lambda a, b: a + b
)
print("Total age:", round(total_age, 2))

print("\n6. Average age using RDD operations:")
average_age = total_age / data_lines.count()
print("Average age:", round(average_age, 2))

print("\n7. Partition analysis:")
print("Number of partitions:", data_lines.getNumPartitions())
print("Records per partition:", data_lines.glom().map(len).collect())

print("\n===== RDD ANALYTICS COMPLETED =====")

spark.stop()

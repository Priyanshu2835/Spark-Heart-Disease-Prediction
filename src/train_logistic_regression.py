
import glob
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

spark = (
    SparkSession.builder
    .appName("INT315_LogisticRegression")
    .master("local[*]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

files = glob.glob("data/heart_disease_clean.csv/*.csv")
if not files:
    raise FileNotFoundError(
        "Cleaned dataset not found. Run preprocessing.py first."
    )

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(files[0])
)

# Original num is retained for analysis, but excluded from features.
# label is the binary target.
categorical_cols = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]
numeric_cols = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

# Create categorical indices and one-hot vectors.
indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_idx",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=[c + "_idx" for c in categorical_cols],
    outputCols=[c + "_vec" for c in categorical_cols],
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=numeric_cols + [c + "_vec" for c in categorical_cols],
    outputCol="raw_features",
    handleInvalid="error"
)

scaler = StandardScaler(
    inputCol="raw_features",
    outputCol="features",
    withMean=False,
    withStd=True
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, lr])

# Stratification isn't guaranteed by randomSplit, so use a fixed seed.
train, test = df.randomSplit([0.8, 0.2], seed=42)

print("\n===== LOGISTIC REGRESSION TRAINING =====")
print("Total records:", df.count())
print("Training records:", train.count())
print("Testing records:", test.count())

model = pipeline.fit(train)
predictions = model.transform(test).cache()

print("\n===== SAMPLE PREDICTIONS =====")
predictions.select(
    "label", "prediction", "probability"
).show(10, truncate=False)

# Binary classification metrics.
binary_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = binary_eval.evaluate(predictions)

# Weighted metrics.
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

print("\n===== TEST METRICS =====")
print("Accuracy:", round(accuracy_eval.evaluate(predictions), 4))
print("Weighted precision:", round(precision_eval.evaluate(predictions), 4))
print("Weighted recall:", round(recall_eval.evaluate(predictions), 4))
print("Weighted F1:", round(f1_eval.evaluate(predictions), 4))
print("Area under ROC:", round(auc, 4))

print("\n===== CONFUSION MATRIX =====")
predictions.groupBy("label", "prediction").count().orderBy(
    "label", "prediction"
).show()

# Save the fitted Spark ML pipeline.
model_path = "models/logistic_regression_pipeline"
model.write().overwrite().save(model_path)
print("Saved model to:", model_path)

predictions.unpersist()
spark.stop()

print("\n===== LOGISTIC REGRESSION COMPLETED =====")

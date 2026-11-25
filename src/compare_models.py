
import csv
import glob
import os

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

spark = (
    SparkSession.builder
    .appName("INT315_Model_Comparison")
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

categorical_cols = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]
numeric_cols = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

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
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=False,
    withStd=True
)

lr = LogisticRegression(
    featuresCol="scaledFeatures",
    labelCol="label",
    maxIter=100,
    regParam=0.01,
    elasticNetParam=0.0
)

tree = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    seed=42
)

lr_pipeline = Pipeline(
    stages=indexers + [encoder, assembler, scaler, lr]
)

tree_pipeline = Pipeline(
    stages=indexers + [encoder, assembler, tree]
)

train, test = df.randomSplit([0.8, 0.2], seed=42)

print("\n===== MODEL COMPARISON =====")
print("Total records:", df.count())
print("Training records:", train.count())
print("Testing records:", test.count())

models = [
    ("Logistic Regression", lr_pipeline),
    ("Decision Tree", tree_pipeline)
]

results = []

accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
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
f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)
auc_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

for name, pipeline in models:
    print("\nTraining:", name)

    fitted_model = pipeline.fit(train)
    predictions = fitted_model.transform(test).cache()

    metrics = {
        "Model": name,
        "Accuracy": accuracy_eval.evaluate(predictions),
        "Weighted Precision": precision_eval.evaluate(predictions),
        "Weighted Recall": recall_eval.evaluate(predictions),
        "Weighted F1": f1_eval.evaluate(predictions),
        "ROC-AUC": auc_eval.evaluate(predictions)
    }
    results.append(metrics)

    print("\nMetrics for", name)
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nConfusion matrix:")
    predictions.groupBy("label", "prediction").count().orderBy(
        "label", "prediction"
    ).show()

    predictions.unpersist()

os.makedirs("outputs", exist_ok=True)
csv_path = "outputs/model_comparison.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    for result in results:
        writer.writerow({
            key: value if key == "Model" else round(value, 4)
            for key, value in result.items()
        })

print("\n===== FINAL COMPARISON =====")
print(
    f"{'Model':<22} {'Accuracy':>10} {'Precision':>10} "
    f"{'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
)

for result in results:
    print(
        f"{result['Model']:<22} "
        f"{result['Accuracy']:>10.4f} "
        f"{result['Weighted Precision']:>10.4f} "
        f"{result['Weighted Recall']:>10.4f} "
        f"{result['Weighted F1']:>10.4f} "
        f"{result['ROC-AUC']:>10.4f}"
    )

print("\nComparison saved to:", csv_path)

spark.stop()
print("\n===== COMPARISON COMPLETED =====")

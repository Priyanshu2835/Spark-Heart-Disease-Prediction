
import glob

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

spark = (
    SparkSession.builder
    .appName("INT315_DecisionTree")
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

tree = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=5,
    seed=42
)

pipeline = Pipeline(stages=indexers + [encoder, assembler, tree])

train, test = df.randomSplit([0.8, 0.2], seed=42)

print("\n===== DECISION TREE TRAINING =====")
print("Total records:", df.count())
print("Training records:", train.count())
print("Testing records:", test.count())

model = pipeline.fit(train)
predictions = model.transform(test).cache()

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

print("\n===== TEST METRICS =====")
print("Accuracy:", round(accuracy_eval.evaluate(predictions), 4))
print("Weighted precision:", round(precision_eval.evaluate(predictions), 4))
print("Weighted recall:", round(recall_eval.evaluate(predictions), 4))
print("Weighted F1:", round(f1_eval.evaluate(predictions), 4))
print("Area under ROC:", round(auc_eval.evaluate(predictions), 4))

print("\n===== CONFUSION MATRIX =====")
predictions.groupBy("label", "prediction").count().orderBy(
    "label", "prediction"
).show()

model_path = "models/decision_tree_pipeline"
model.write().overwrite().save(model_path)
print("Saved model to:", model_path)

predictions.unpersist()
spark.stop()

print("\n===== DECISION TREE COMPLETED =====")


import os

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    PROJECT_DIR, "models", "logistic_regression_pipeline"
)

# Each feature uses the same column name as model training.
FIELDS = [
    ("age", "Age in years", 1, 120),
    ("sex", "Sex (0 or 1)", 0, 1),
    ("cp", "Chest pain type (1-4)", 1, 4),
    ("trestbps", "Resting blood pressure", 50, 300),
    ("chol", "Serum cholesterol", 50, 700),
    ("fbs", "Fasting blood sugar > 120 mg/dl (0 or 1)", 0, 1),
    ("restecg", "Resting ECG result (0-2)", 0, 2),
    ("thalach", "Maximum heart rate achieved", 40, 250),
    ("exang", "Exercise-induced angina (0 or 1)", 0, 1),
    ("oldpeak", "Oldpeak", 0, 10),
    ("slope", "Slope of peak exercise ST segment (1-3)", 1, 3),
    ("ca", "Number of major vessels (0-3)", 0, 3),
    ("thal", "Thal category (3, 6 or 7)", 3, 7),
]

CATEGORICAL_ALLOWED = {
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "exang": {0, 1},
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def read_value(name, description, minimum, maximum):
    while True:
        raw = input(f"{description}: ").strip()

        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if not minimum <= value <= maximum:
            print(f"Enter a value between {minimum} and {maximum}.")
            continue

        if name in CATEGORICAL_ALLOWED:
            if value != int(value) or int(value) not in CATEGORICAL_ALLOWED[name]:
                allowed = sorted(CATEGORICAL_ALLOWED[name])
                print("Allowed values:", allowed)
                continue

        return value


def main():
    if not os.path.isdir(MODEL_PATH):
        print("Saved Logistic Regression model was not found.")
        print("Run option 5 in the main menu to train and save it first.")
        return

    print("\n===== PATIENT RECORD PREDICTION =====")
    print("Enter values using the dataset's coding scheme.")
    print("This is an academic demonstration, not medical advice.")

    record = {}
    for name, description, minimum, maximum in FIELDS:
        record[name] = read_value(
            name, description, minimum, maximum
        )

    spark = (
        SparkSession.builder
        .appName("INT315_Patient_Prediction")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        model = PipelineModel.load(MODEL_PATH)

        # The model was trained on these feature columns.
        # No diagnosis label or original target column is supplied.
        patient_df = spark.createDataFrame([record])
        prediction = model.transform(patient_df).first()

        predicted_class = int(prediction["prediction"])
        probability = prediction["probability"]

        print("\n===== PREDICTION RESULT =====")
        print("Predicted class:", predicted_class)
        print(
            "Model probability for class 0:",
            round(float(probability[0]), 4)
        )
        print(
            "Model probability for class 1:",
            round(float(probability[1]), 4)
        )

        if predicted_class == 1:
            print("Model output: Positive class")
        else:
            print("Model output: Negative class")

        print(
            "\nImportant: This output is not a clinical diagnosis "
            "or a validated individual risk estimate."
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

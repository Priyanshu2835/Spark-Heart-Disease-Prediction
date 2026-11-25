
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MENU = {
    "1": (
        "Dataset overview and missing-value inspection",
        "inspect_data.py"
    ),
    "2": ("Data preprocessing", "preprocessing.py"),
    "3": ("RDD operations and partition analysis", "rdd_analytics.py"),
    "4": ("Spark SQL healthcare analytics", "sql_analytics.py"),
    "5": ("Train Logistic Regression", "train_logistic_regression.py"),
    "6": ("Train Decision Tree", "train_decision_tree.py"),
    "7": ("Compare model performance", "compare_models.py"),
    "8": ("Correlation analysis", "correlation_analysis.py"),
    "9": ("Benchmark Spark execution", "benchmark_spark.py"),
    "10": ("Predict from a patient record", "predict_patient.py"),
}


def run_module(script_name):
    script_path = os.path.join(
        PROJECT_DIR, "src", script_name
    )

    if not os.path.isfile(script_path):
        print("\nERROR: Script not found:", script_path)
        return

    print("\n" + "=" * 65)
    print("RUNNING:", script_name)
    print("=" * 65)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_DIR,
            check=False
        )

        print("\n" + "-" * 65)

        if result.returncode == 0:
            print("Module completed successfully.")
        else:
            print("Module exited with return code:", result.returncode)
            print("Review the error output above.")

        print("-" * 65)

    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")


def main():
    while True:
        print("\n" + "=" * 65)
        print(" DISTRIBUTED HEALTHCARE ANALYTICS")
        print(" Apache Spark | PySpark | Spark MLlib")
        print("=" * 65)

        for option, (description, _) in MENU.items():
            print(f"{option}. {description}")

        print("0. Exit")

        choice = input("\nEnter your choice: ").strip()

        if choice == "0":
            print("\nThank you for using Distributed Healthcare Analytics.")
            break

        elif choice in MENU:
            _, script_name = MENU[choice]
            run_module(script_name)
            input("\nPress Enter to return to the main menu...")

        else:
            print("\nInvalid choice. Please select an option from 0 to 9.")


if __name__ == "__main__":
    main()

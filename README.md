\# Distributed Healthcare Analytics and Heart Disease Risk Prediction Using Apache Spark and PySpark



\## 1. Project Overview



This project demonstrates distributed data processing and healthcare analytics using Apache Spark and PySpark. It uses a heart disease dataset to perform data inspection, missing-value handling, RDD operations, Spark SQL analysis, correlation analysis, machine learning, model comparison, execution benchmarking, and interactive terminal-based prediction.



This is an academic demonstration. Predictions must not be used as medical diagnoses or for clinical decision-making.



\## 2. Objectives



\- Inspect the dataset schema, record count, and missing values.

\- Clean missing feature values using median imputation.

\- Demonstrate Spark Core RDD transformations and actions.

\- Perform healthcare analytics using Spark SQL.

\- Train Logistic Regression and Decision Tree classifiers using Spark MLlib.

\- Evaluate and compare classification performance.

\- Examine correlations between selected numeric features and the target.

\- Compare execution times under different partition configurations.

\- Predict the class for a user-entered patient record.



\## 3. Technologies Used



\- Python 3.11.9

\- Apache Spark 3.5.3

\- PySpark 3.5.3

\- Spark Core RDD

\- Spark SQL

\- Spark MLlib

\- Java JDK 21

\- Windows Command Prompt



\## 4. Dataset



\*\*Dataset:\*\* UCI Heart Disease — processed Cleveland dataset



\*\*Official source:\*\* https://archive.ics.uci.edu/dataset/45/heart+disease



The original processed Cleveland dataset contains 303 records and 14 columns. The target column `num` represents the original diagnosis category. For binary classification, the project maps `num = 0` to class 0 and `num > 0` to class 1.



The original target distribution is:



\- Class 0: 164 records

\- Class 1: 139 records after binary conversion



The original data contains missing values represented by `?`. The preprocessing module handles missing feature values and produces a cleaned dataset with 303 records, 15 columns, and no remaining missing values.



\## 5. Project Structure



```text

INT315\_Healthcare\_Spark/

├── data/

│   ├── processed.cleveland.data

│   └── heart\_disease\_clean.csv/

├── models/

│   ├── logistic\_regression\_pipeline/

│   └── decision\_tree\_pipeline/

├── outputs/

│   ├── model\_comparison.csv

│   ├── correlation\_matrix.csv

│   └── spark\_benchmark.csv

├── src/

│   ├── main.py

│   ├── inspect\_data.py

│   ├── preprocessing.py

│   ├── rdd\_analytics.py

│   ├── sql\_analytics.py

│   ├── train\_logistic\_regression.py

│   ├── train\_decision\_tree.py

│   ├── compare\_models.py

│   ├── correlation\_analysis.py

│   ├── benchmark\_spark.py

│   └── predict\_patient.py

├── README.md

└── requirements.txt

```



Spark may save CSV output as a directory containing part files and a `\_SUCCESS` marker rather than as a single CSV file.



\## 6. Installation and Configuration



Install Python 3.11, Java JDK 21, Apache Spark 3.5.3, and the required Python package.



From Command Prompt:



```cmd

cd /d D:\\INT315\_Healthcare\_Spark

py -3.11 -m pip install -r requirements.txt

```



Verify PySpark:



```cmd

py -3.11 -m pip show pyspark

```



Ensure `JAVA\_HOME` points to the installed JDK and that the Spark installation is configured correctly for the local Windows environment.



\## 7. Running the Project



Launch the terminal-based application:



```cmd

cd /d D:\\INT315\_Healthcare\_Spark

py -3.11 src\\main.py

```



The main menu provides these operations:



1\. Dataset overview and missing-value inspection

2\. Data preprocessing

3\. RDD operations and partition analysis

4\. Spark SQL healthcare analytics

5\. Train Logistic Regression

6\. Train Decision Tree

7\. Compare model performance

8\. Correlation analysis

9\. Benchmark Spark execution

10\. Predict from a patient record

0\. Exit



Run preprocessing before training models if the cleaned dataset needs to be regenerated. Train the Logistic Regression model before using patient prediction if its saved model is unavailable.



\## 8. Implementation Summary



\### Data Inspection and Preprocessing



The inspection module reports the dataset dimensions, schema, sample records, missing-value counts, and original diagnosis distribution. The preprocessing module imputes missing feature values using medians and converts the original diagnosis into a binary target named `label`.



\### RDD Analytics



The RDD module demonstrates record counting, mapping, filtering, key-based aggregation, reduction, and partition analysis.



Verified results include 303 records, 91 records with age greater than or equal to 60, and an average age of approximately 54.44 years. The local execution reported two partitions containing 151 and 152 records.



\### Spark SQL Analytics



Spark SQL queries calculate aggregate statistics, target-class counts, age-group summaries, and comparisons between the two binary classes.



\### Machine Learning



The Logistic Regression and Decision Tree pipelines use categorical feature indexing, one-hot encoding, feature assembly, and classification. A seeded 80:20 random split is used for model evaluation.



\### Correlation Analysis



Pearson correlations are calculated for selected numeric features and the binary target. Correlation describes linear association and does not establish causation.



\### Execution Benchmarking



The benchmarking module compares a healthcare aggregation under different partition counts. Since the dataset is small and the measurements are local, execution times may vary between runs.



\### Patient Prediction



The prediction module collects feature values through the command line, loads the saved Logistic Regression pipeline, and displays the predicted class and model probabilities. The output is for educational purposes only.



\## 9. Verified Results



The project was tested using the local Windows environment.



| Metric | Logistic Regression | Decision Tree |

|---|---:|---:|

| Accuracy | 0.8511 | 0.7021 |

| Weighted precision | 0.8542 | 0.7002 |

| Weighted recall | 0.8511 | 0.7021 |

| Weighted F1-score | 0.8489 | 0.6959 |

| ROC AUC | 0.9111 | 0.5648 |



These values are from the recorded seeded split of 256 training records and 47 test records. The test set is small and not stratified, so the metrics should be interpreted cautiously and do not establish clinical validity.



\## 10. Sample Prediction Test



A test record was entered through the command-line prediction module. The model returned:



\- Predicted class: 1

\- Model probability for class 0: 0.0946

\- Model probability for class 1: 0.9054



These values represent the model's output for that input, not a diagnosis or validated individual risk estimate.



\## 11. Limitations



\- The dataset contains only 303 records.

\- The test set contains 47 records, so individual test cases can substantially affect reported metrics.

\- The random split is not stratified.

\- Results from local partition benchmarks are affected by startup overhead, caching, and machine conditions.

\- The model has not been clinically validated and should not be used to make healthcare decisions.



\## 12. Conclusion



The project demonstrates an end-to-end local Apache Spark workflow, from raw healthcare data inspection and preprocessing to distributed-style analytics, classification, evaluation, benchmarking, and terminal-based prediction. Logistic Regression performed better than the Decision Tree on the recorded evaluation split. Further work could include cross-validation, stratified evaluation, larger datasets, and external validation.



\## 13. References



1\. UCI Machine Learning Repository, Heart Disease Dataset: https://archive.ics.uci.edu/dataset/45/heart+disease

2\. Apache Spark Documentation: https://spark.apache.org/docs/3.5.3/

3\. PySpark Documentation: https://spark.apache.org/docs/3.5.3/api/python/




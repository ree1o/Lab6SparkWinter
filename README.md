# Lab 6: Spark ML Pipeline on Amazon EMR

**Course:** Distributed Computing
**Student:** Kenzhekeyev Runar (IT-2303)
**Platform:** Amazon EMR (Spark on YARN)

---

## 1. Project Overview

This lab implements an **end-to-end Spark ML pipeline** for **customer churn prediction** using distributed processing on Amazon EMR.

The pipeline performs:

* Distributed data loading from HDFS
* Feature engineering (categorical encoding + scaling)
* Feature vector assembly
* Model training
* Prediction generation
* Model evaluation
* Experimental analysis

The goal is to demonstrate how machine learning workflows scale using Spark’s distributed execution model.

---

## 2. Dataset Description

**Dataset:** Bank Customer Churn Dataset
**Source:** Kaggle
[https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)

The dataset contains bank customer information used to predict whether a customer will leave the bank.

### Target Variable

* **Exited**

  * 0 → Customer stayed
  * 1 → Customer churned

### Features Used

#### Categorical

* Geography
* Gender

#### Numerical

* CreditScore
* Age
* Tenure
* Balance
* NumOfProducts
* EstimatedSalary

These features describe customer demographics, account activity, and financial status, which influence churn behavior.

---

## 3. EMR Cluster Configuration

The Spark pipeline was executed on Amazon EMR with:

* 1 Primary (Master) node
* 2 Core nodes
* Applications:

  * Hadoop
  * Spark
* Instance type: m4.large
* Execution mode: YARN cluster processing

---

## 4. Upload Dataset to EMR

### Step 1 — Copy dataset to EMR master

Run from local machine:

```bash
scp -i key.pem Churn_Modelling.csv hadoop@<master-public-dns>:/home/hadoop/
```

### Step 2 — Upload to HDFS

```bash
hdfs dfs -mkdir -p /user/hadoop/churn_input
hdfs dfs -put -f /home/hadoop/Churn_Modelling.csv /user/hadoop/churn_input/
hdfs dfs -ls /user/hadoop/churn_input
```

---

## 5. Spark ML Pipeline Stages

The pipeline follows Spark ML design principles and includes:

1. Data loading from HDFS
2. String indexing (categorical encoding)
3. One-hot encoding
4. Feature vector assembly
5. Feature scaling
6. Model training
7. Prediction
8. Evaluation

### Key Spark Components Used

* `StringIndexer`
* `OneHotEncoder`
* `VectorAssembler`
* `StandardScaler`
* `LogisticRegression`
* `RandomForestClassifier`
* `Pipeline`

---

## 6. Running the Spark Job

### Submit job on EMR

```bash
spark-submit \
 --master yarn \
 --deploy-mode client \
 churn_pipeline.py --experiment none
```

---

## 7. Experiments

### Option B — Feature Ablation

Removes categorical features to evaluate their impact on model performance.

Run:

```bash
spark-submit \
 --master yarn \
 --deploy-mode client \
 churn_pipeline.py --experiment ablation
```

Compares:

* Accuracy with categorical features
* Accuracy without categorical features
* Runtime differences

---

### Option C — Model Comparison

Compares Logistic Regression vs Random Forest.

Run:

```bash
spark-submit \
 --master yarn \
 --deploy-mode client \
 churn_pipeline.py --experiment model
```

Evaluates:

* Prediction accuracy
* Training runtime
* Model performance differences

---

## 8. Monitoring Distributed Execution

Spark jobs were monitored via YARN ResourceManager:

```
http://<master-public-dns>:8088
```

Observed metrics:

* Number of executors
* Tasks per stage
* Distributed job execution

---

## 9. Repository Structure

```
Lab6_Spark_Churn/
 ├── churn_pipeline.py
 └── README.md
```

---

## 10. Key Learning Outcomes

* Built a distributed ML pipeline using Spark ML
* Performed large-scale feature engineering
* Trained models on a distributed cluster
* Evaluated model accuracy
* Compared ML performance through experimentation
* Understood scalability of ML workloads on EMR

---

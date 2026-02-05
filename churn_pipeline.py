#!/usr/bin/env python3
import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def build_pipeline(include_categorical: bool, model_type: str):
    # Categorical processing
    geo_indexer = StringIndexer(
        inputCol="Geography", outputCol="GeographyIndex", handleInvalid="keep"
    )
    gender_indexer = StringIndexer(
        inputCol="Gender", outputCol="GenderIndex", handleInvalid="keep"
    )
    encoder = OneHotEncoder(
        inputCols=["GeographyIndex", "GenderIndex"],
        outputCols=["GeographyVec", "GenderVec"]
    )

    # Feature columns
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    cat_cols = ["GeographyVec", "GenderVec"]

    assembler_inputs = numeric_cols + (cat_cols if include_categorical else [])
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

    if model_type == "lr":
        clf = LogisticRegression(labelCol="Exited", featuresCol="scaledFeatures", maxIter=30)
    elif model_type == "rf":
        clf = RandomForestClassifier(labelCol="Exited", featuresCol="scaledFeatures", numTrees=100, maxDepth=8)
    else:
        raise ValueError("model_type must be 'lr' or 'rf'")

    stages = [geo_indexer, gender_indexer, encoder, assembler, scaler, clf]
    return Pipeline(stages=stages)


def run_one(spark, data, include_categorical: bool, model_type: str, seed: int):
    train, test = data.randomSplit([0.8, 0.2], seed=seed)

    pipeline = build_pipeline(include_categorical=include_categorical, model_type=model_type)

    t0 = time.time()
    model = pipeline.fit(train)
    preds = model.transform(test)
    # Force execution so timing is real (Spark is lazy)
    preds.cache()
    preds.count()
    elapsed = time.time() - t0

    evaluator = MulticlassClassificationEvaluator(
        labelCol="Exited", predictionCol="prediction", metricName="accuracy"
    )
    acc = evaluator.evaluate(preds)

    return acc, elapsed, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["none", "ablation", "model"], default="ablation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spark = SparkSession.builder.appName("CustomerChurnPipeline").getOrCreate()

    data = spark.read.csv(
        "hdfs:///user/hadoop/churn_input/Churn_Modelling.csv",
        header=True,
        inferSchema=True
    )

    # Keep only columns we need; drop IDs/names to avoid leakage/noise
    data = data.select(
        "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
        "NumOfProducts", "EstimatedSalary", "Exited"
    ).withColumn("Exited", col("Exited").cast("double"))

    print("\n=== Data Sample ===")
    data.show(5, truncate=False)

    if args.experiment == "none":
        acc, sec, preds = run_one(spark, data, include_categorical=True, model_type="lr", seed=args.seed)
        print(f"\n[BASELINE] LogisticRegression + categorical features")
        print(f"Accuracy = {acc:.4f}, Runtime = {sec:.2f}s")
        print("\nSample predictions:")
        preds.select("Exited", "prediction", "probability").show(10, truncate=False)

    elif args.experiment == "ablation":
        acc_full, sec_full, _ = run_one(spark, data, include_categorical=True, model_type="lr", seed=args.seed)
        acc_no_cat, sec_no_cat, _ = run_one(spark, data, include_categorical=False, model_type="lr", seed=args.seed)

        print("\n[EXPERIMENT B: Feature Ablation]")
        print("Compare LogisticRegression WITH vs WITHOUT categorical features")
        print(f"WITH categorical:    Accuracy = {acc_full:.4f}, Runtime = {sec_full:.2f}s")
        print(f"WITHOUT categorical: Accuracy = {acc_no_cat:.4f}, Runtime = {sec_no_cat:.2f}s")

    elif args.experiment == "model":
        acc_lr, sec_lr, _ = run_one(spark, data, include_categorical=True, model_type="lr", seed=args.seed)
        acc_rf, sec_rf, _ = run_one(spark, data, include_categorical=True, model_type="rf", seed=args.seed)

        print("\n[EXPERIMENT C: Model Comparison]")
        print("Compare LogisticRegression vs RandomForest (same features)")
        print(f"LogisticRegression: Accuracy = {acc_lr:.4f}, Runtime = {sec_lr:.2f}s")
        print(f"RandomForest:       Accuracy = {acc_rf:.4f}, Runtime = {sec_rf:.2f}s")

    spark.stop()


if __name__ == "__main__":
    main()

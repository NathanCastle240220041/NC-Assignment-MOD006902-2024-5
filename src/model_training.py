from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

def train_models(processed_df: DataFrame):
    """
    Train multiple classifiers on the processed DataFrame.

    Args:
        processed_df (DataFrame): Preprocessed Spark DataFrame with 'features' and 'label'.

    Returns:
        dict: Dictionary of trained models and their test predictions.
    """
    # Split data
    train_data, test_data = processed_df.randomSplit([0.8, 0.2], seed=42)

    models = {
        "LogisticRegression": LogisticRegression(featuresCol="features", labelCol="label"),
        "NaiveBayes": NaiveBayes(featuresCol="features", labelCol="label"),
        "RandomForest": RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=100),
    }

    results = {}

    for name, model in models.items():
        clf_pipeline = Pipeline(stages=[model])
        fitted_model = clf_pipeline.fit(train_data)
        predictions = fitted_model.transform(test_data)
        results[name] = {
            "model": fitted_model,
            "predictions": predictions,
        }

    return results

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame
from pyspark.mllib.evaluation import MulticlassMetrics


def evaluate_model(model_pipeline, test_data: DataFrame, label_col="label"):
    """
    Evaluates the given model pipeline on test data and returns key metrics.

    Args:
        model_pipeline: Trained PySpark PipelineModel.
        test_data (DataFrame): Preprocessed test data.
        label_col (str): Name of the label column.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    # Generate predictions
    predictions = model_pipeline.transform(test_data)

    # Select prediction and label
    prediction_and_labels = predictions.select("prediction", label_col)

    # Convert to RDD for MulticlassMetrics
    rdd = prediction_and_labels.rdd.map(lambda row: (float(row[0]), float(row[1])))
    metrics = MulticlassMetrics(rdd)

    # Accuracy, precision, recall, f1
    accuracy = metrics.accuracy
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1_score = metrics.fMeasure(1.0)

    print("\nModel Evaluation Results:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1_score:.4f}\n")

    # Confusion matrix
    print("Confusion Matrix:")
    print(metrics.confusionMatrix())

    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

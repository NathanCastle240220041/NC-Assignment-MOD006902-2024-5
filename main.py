from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model_training import train_models
from src.evaluation import evaluate_model
from src.graphs import (
    plot_top_tfidf_words,
    plot_logistic_regression_weights,
    plot_random_forest_importance,
    plot_naive_bayes_indicative_words
)

if __name__ == "__main__":
    # Load raw data
    df = load_data("training.1600000.processed.noemoticon.csv", sample_fraction=0.1)
    df.show(5, truncate=80)

    # Preprocess data
    processed_df, pipeline_model = preprocess_data(df)
    processed_df.show(5)

    # Train models
    trained_models = train_models(processed_df)  # Returns dict of name: pipeline_model

    # Evaluate models
    print("\n================ Model Evaluation ================")
    for name, model in trained_models.items():
        print(f"\n--- Evaluating: {name} ---")
        evaluate_model(model, processed_df)

    # Extract TF-IDF vocabulary from the shared pipeline
    cv_model = pipeline_model.stages[2]  # CountVectorizer is 3rd in pipeline

    print("\n================ Graphical Analysis ================")
    plot_top_tfidf_words(pipeline_model, df)

    if "Logistic Regression" in trained_models:
        plot_logistic_regression_weights(trained_models["Logistic Regression"], cv_model)

    if "Random Forest" in trained_models:
        plot_random_forest_importance(trained_models["Random Forest"], cv_model)

    if "Naive Bayes" in trained_models:
        plot_naive_bayes_indicative_words(trained_models["Naive Bayes"], cv_model)

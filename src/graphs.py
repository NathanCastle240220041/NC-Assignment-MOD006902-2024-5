import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml import Pipeline

def plot_top_tfidf_words(df, top_n=20):
    """
    Generates a bar chart of the top N TF-IDF weighted words in the given DataFrame.

    Args:
        df (DataFrame): Input Spark DataFrame with a 'text' column.
        top_n (int): Number of top TF-IDF words to display.

    Returns:
        None. Displays a horizontal bar chart using matplotlib.
    """
    # Define preprocessing pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    cv = CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=20000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf])
    model = pipeline.fit(df)
    tfidf_df = model.transform(df)

    # Extract vocabulary and feature weights
    vocab = model.stages[2].vocabulary
    weights = tfidf_df.select("features").limit(1).collect()[0]["features"].toArray()
    zipped = list(zip(vocab, weights))
    top_words = sorted(zipped, key=lambda x: x[1], reverse=True)[:top_n]

    # Plot results
    labels, scores = zip(*top_words[::-1])
    plt.figure(figsize=(10, 5))
    plt.barh(labels, scores)
    plt.title("Top TF-IDF Words")
    plt.xlabel("TF-IDF Score")
    plt.tight_layout()
    plt.show()

def plot_logreg_weights(train_df):
    """
    Trains a Logistic Regression model and visualizes top positive and negative indicative words.

    Args:
        train_df (DataFrame): Input DataFrame with 'text' and 'label' columns.

    Returns:
        None. Displays two horizontal bar charts.
    """
    # Define pipeline
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered"),
        CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=10000),
        IDF(inputCol="rawFeatures", outputCol="features")
    ])
    model = pipeline.fit(train_df)
    features_df = model.transform(train_df)
    cv_model = model.stages[2]
    vocab = cv_model.vocabulary

    # Train Logistic Regression
    lr = LogisticRegression(labelCol="label", featuresCol="features")
    lr_model = lr.fit(features_df)
    weights = list(zip(vocab, lr_model.coefficients.toArray()))
    sorted_weights = sorted(weights, key=lambda x: x[1])

    # Plot top 10 negative and positive words
    top_neg = sorted_weights[:10]
    top_pos = sorted_weights[-10:]
    labels_neg, scores_neg = zip(*top_neg)
    labels_pos, scores_pos = zip(*top_pos)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.barh(labels_neg, scores_neg)
    plt.title("LogReg: Top Negative Words")

    plt.subplot(1, 2, 2)
    plt.barh(labels_pos, scores_pos)
    plt.title("LogReg: Top Positive Words")
    plt.tight_layout()
    plt.show()

def plot_rf_feature_importance(train_df):
    """
    Trains a Random Forest model and plots the top 10 most important words.

    Args:
        train_df (DataFrame): Input DataFrame with 'text' and 'label' columns.

    Returns:
        None. Displays a horizontal bar chart.
    """
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered"),
        CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=10000),
        IDF(inputCol="rawFeatures", outputCol="features")
    ])
    model = pipeline.fit(train_df)
    features_df = model.transform(train_df)
    cv_model = model.stages[2]
    vocab = cv_model.vocabulary

    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100)
    rf_model = rf.fit(features_df)
    importances = rf_model.featureImportances.toArray()

    top_rf = sorted(zip(vocab, importances), key=lambda x: x[1], reverse=True)[:10]
    labels, scores = zip(*top_rf[::-1])

    plt.figure(figsize=(8, 4))
    plt.barh(labels, scores)
    plt.title("Random Forest: Top Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

def plot_nb_informative_words(train_df):
    """
    Trains a Naive Bayes model and visualizes the words most indicative of positive vs negative classes.

    Args:
        train_df (DataFrame): Input DataFrame with 'text' and 'label' columns.

    Returns:
        None. Displays two horizontal bar charts.
    """
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="text", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="filtered"),
        CountVectorizer(inputCol="filtered", outputCol="rawFeatures", vocabSize=10000),
        IDF(inputCol="rawFeatures", outputCol="features")
    ])
    model = pipeline.fit(train_df)
    features_df = model.transform(train_df)
    cv_model = model.stages[2]
    vocab = cv_model.vocabulary

    nb = NaiveBayes(labelCol="label", featuresCol="features")
    nb_model = nb.fit(features_df)
    word_diffs = nb_model.theta.toArray()[1] - nb_model.theta.toArray()[0]
    weights = list(zip(vocab, word_diffs))
    sorted_weights = sorted(weights, key=lambda x: x[1])

    # Plot most negative and most positive indicative words
    top_neg = sorted_weights[:10]
    top_pos = sorted_weights[-10:]
    labels_neg, scores_neg = zip(*top_neg)
    labels_pos, scores_pos = zip(*top_pos)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.barh(labels_neg, scores_neg)
    plt.title("Naive Bayes: Negative Words")

    plt.subplot(1, 2, 2)
    plt.barh(labels_pos, scores_pos)
    plt.title("Naive Bayes: Positive Words")
    plt.tight_layout()
    plt.show()

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import lower, col


def preprocess_data(df, num_features=10000):
    """
    Preprocesses the tweet DataFrame into features for ML.

    Steps:
    - Lowercase text
    - Tokenize using RegexTokenizer
    - Remove stopwords
    - Convert to TF
    - Apply IDF weighting

    Args:
        df (DataFrame): Input DataFrame with 'text' and 'label'.
        num_features (int): Number of features to hash in TF.

    Returns:
        transformed_df (DataFrame): DataFrame with 'label' and 'features' columns.
        pipeline_model (PipelineModel): Trained pipeline model.
    """

    # Step 1: Lowercase all text
    df_clean = df.withColumn("text", lower(col("text")))

    # Step 2: Tokenize (regex = words only)
    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W")

    # Step 3: Remove stopwords
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")

    # Step 4: Term Frequency (TF)
    hashing_tf = HashingTF(inputCol="filtered_tokens", outputCol="raw_features", numFeatures=num_features)

    # Step 5: Inverse Document Frequency (IDF)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # Create the pipeline
    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])
    pipeline_model = pipeline.fit(df_clean)
    transformed_df = pipeline_model.transform(df_clean)

    # Return only necessary columns
    return transformed_df.select("label", "features"), pipeline_model

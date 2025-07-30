from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col, when


def load_data(file_path: str, sample_fraction: float = 0.1, seed: int = 42):
    """
    Loads the Sentiment140 dataset and returns a preprocessed Spark DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        sample_fraction (float): Fraction of data to sample (for speed). Default is 0.1 (10%).
        seed (int): Random seed for sampling.

    Returns:
        pyspark.sql.DataFrame: A DataFrame with 'text' and binary 'label' columns.
    """
    spark = SparkSession.builder.appName("Sentiment140Loader").getOrCreate()

    schema = StructType([
        StructField("target", IntegerType(), True),
        StructField("ids", StringType(), True),
        StructField("date", StringType(), True),
        StructField("flag", StringType(), True),
        StructField("user", StringType(), True),
        StructField("text", StringType(), True),
    ])

    df = spark.read.csv(file_path, schema=schema)

    # Filter only positive (4) and negative (0) examples
    df_filtered = df.filter((col("target") == 0) | (col("target") == 4))

    # Convert labels: 4 → 1 (positive), 0 → 0 (negative)
    df_filtered = df_filtered.withColumn("label", when(col("target") == 4, 1).otherwise(0))

    # Select necessary columns
    df_final = df_filtered.select("text", "label")

    # Optional: downsample for speed
    if sample_fraction < 1.0:
        df_final = df_final.sample(withReplacement=False, fraction=sample_fraction, seed=seed)

    return df_final

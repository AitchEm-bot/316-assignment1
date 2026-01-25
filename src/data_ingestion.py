"""
Data Ingestion Module
Handles loading raw data into Spark DataFrames with proper schema.
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, DateType, TimestampType
)


def create_spark_session(app_name="bigboyz-dubai-real-estate"):
    """
    Creates and returns a configured Spark session.

    Args:
        app_name: Name for the Spark application

    Returns:
        SparkSession: Configured Spark session
    """
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

    # Set log level to reduce noise
    spark.sparkContext.setLogLevel("WARN")

    return spark


def get_transactions_schema():
    """
    Returns the schema for the Dubai Land Department transactions dataset.
    Based on actual Transactions.csv from Dubai Pulse.

    Returns:
        StructType: Schema definition for transactions data
    """
    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("procedure_id", StringType(), True),
        StructField("trans_group_id", StringType(), True),
        StructField("trans_group_ar", StringType(), True),
        StructField("trans_group_en", StringType(), True),
        StructField("procedure_name_ar", StringType(), True),
        StructField("procedure_name_en", StringType(), True),
        StructField("instance_date", StringType(), True),
        StructField("property_type_id", StringType(), True),
        StructField("property_type_ar", StringType(), True),
        StructField("property_type_en", StringType(), True),
        StructField("property_sub_type_id", StringType(), True),
        StructField("property_sub_type_ar", StringType(), True),
        StructField("property_sub_type_en", StringType(), True),
        StructField("property_usage_ar", StringType(), True),
        StructField("property_usage_en", StringType(), True),
        StructField("reg_type_id", StringType(), True),
        StructField("reg_type_ar", StringType(), True),
        StructField("reg_type_en", StringType(), True),
        StructField("area_id", StringType(), True),
        StructField("area_name_ar", StringType(), True),
        StructField("area_name_en", StringType(), True),
        StructField("building_name_ar", StringType(), True),
        StructField("building_name_en", StringType(), True),
        StructField("project_number", StringType(), True),
        StructField("project_name_ar", StringType(), True),
        StructField("project_name_en", StringType(), True),
        StructField("master_project_en", StringType(), True),
        StructField("master_project_ar", StringType(), True),
        StructField("nearest_landmark_ar", StringType(), True),
        StructField("nearest_landmark_en", StringType(), True),
        StructField("nearest_metro_ar", StringType(), True),
        StructField("nearest_metro_en", StringType(), True),
        StructField("nearest_mall_ar", StringType(), True),
        StructField("nearest_mall_en", StringType(), True),
        StructField("rooms_ar", StringType(), True),
        StructField("rooms_en", StringType(), True),
        StructField("has_parking", IntegerType(), True),
        StructField("procedure_area", DoubleType(), True),
        StructField("actual_worth", DoubleType(), True),
        StructField("meter_sale_price", DoubleType(), True),
        StructField("rent_value", DoubleType(), True),
        StructField("meter_rent_price", DoubleType(), True),
        StructField("no_of_parties_role_1", IntegerType(), True),
        StructField("no_of_parties_role_2", IntegerType(), True),
        StructField("no_of_parties_role_3", IntegerType(), True),
    ])
    return schema


def load_transactions(spark, file_path, use_schema=False):
    """
    Loads the transactions CSV file into a Spark DataFrame.

    Args:
        spark: SparkSession instance
        file_path: Path to the transactions.csv file
        use_schema: If True, use predefined schema; if False, infer schema

    Returns:
        DataFrame: Spark DataFrame containing transaction data
    """
    if use_schema:
        schema = get_transactions_schema()
        df = spark.read.csv(
            file_path,
            header=True,
            schema=schema,
            mode="DROPMALFORMED"
        )
    else:
        # Let Spark infer the schema (slower but safer for initial exploration)
        df = spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            mode="DROPMALFORMED"
        )

    return df


def explore_data(df):
    """
    Performs initial data exploration and prints summary statistics.

    Args:
        df: Spark DataFrame to explore
    """
    print("=" * 60)
    print("DATA EXPLORATION SUMMARY")
    print("=" * 60)

    # Basic info
    print(f"\nTotal Records: {df.count():,}")
    print(f"Number of Columns: {len(df.columns)}")

    # Schema
    print("\n--- Schema ---")
    df.printSchema()

    # Sample data
    print("\n--- Sample Data (5 rows) ---")
    df.show(5, truncate=False)

    # Summary statistics for numeric columns
    print("\n--- Summary Statistics ---")
    df.describe().show()

    # Missing values
    print("\n--- Missing Values per Column ---")
    from pyspark.sql.functions import col, count, when, isnan

    missing_counts = df.select([
        count(when(col(c).isNull() | isnan(col(c)), c)).alias(c)
        if df.schema[c].dataType.typeName() in ['double', 'float', 'integer', 'long']
        else count(when(col(c).isNull(), c)).alias(c)
        for c in df.columns
    ])
    missing_counts.show()

    return df

"""
Feature Engineering Module
Transforms raw features into ML-ready format using Spark MLlib.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, year, month, quarter, dayofweek, dayofmonth,
    to_date, datediff, lit, log1p, sqrt
)
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, MinMaxScaler
)
from pyspark.ml import Pipeline


def extract_temporal_features(df, date_col, date_format='yyyy-MM-dd'):
    """
    Extracts temporal features from a date column.

    Args:
        df: Input Spark DataFrame
        date_col: Name of the date column
        date_format: Format of the date string

    Returns:
        DataFrame: DataFrame with additional temporal features
    """
    # Convert to date type if string
    df = df.withColumn(
        date_col + '_parsed',
        to_date(col(date_col), date_format)
    )

    # Extract temporal features
    df = df.withColumn('trans_year', year(col(date_col + '_parsed')))
    df = df.withColumn('trans_month', month(col(date_col + '_parsed')))
    df = df.withColumn('trans_quarter', quarter(col(date_col + '_parsed')))
    df = df.withColumn('trans_dayofweek', dayofweek(col(date_col + '_parsed')))

    # Drop intermediate column
    df = df.drop(date_col + '_parsed')

    print(f"Extracted temporal features: trans_year, trans_month, trans_quarter, trans_dayofweek")

    return df


def encode_categorical_columns(df, categorical_cols, handle_invalid='keep'):
    """
    Encodes categorical columns using StringIndexer.
    Note: OneHotEncoder can be added if needed, but increases dimensionality.

    Args:
        df: Input Spark DataFrame
        categorical_cols: List of categorical column names
        handle_invalid: How to handle invalid/unseen values ('keep', 'skip', 'error')

    Returns:
        tuple: (transformed DataFrame, list of indexer models)
    """
    indexers = []
    indexed_cols = []

    for col_name in categorical_cols:
        indexed_col = col_name + '_indexed'
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=indexed_col,
            handleInvalid=handle_invalid
        )
        indexers.append(indexer)
        indexed_cols.append(indexed_col)

    # Fit and transform
    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(df)
    df_indexed = model.transform(df)

    print(f"Encoded {len(categorical_cols)} categorical columns")

    return df_indexed, indexed_cols, model


def create_feature_vector(df, numeric_cols, indexed_cols, output_col='features'):
    """
    Assembles all feature columns into a single vector column.

    Args:
        df: Input Spark DataFrame
        numeric_cols: List of numeric feature column names
        indexed_cols: List of indexed categorical column names
        output_col: Name of the output vector column

    Returns:
        DataFrame: DataFrame with features vector column
    """
    all_feature_cols = numeric_cols + indexed_cols

    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol=output_col,
        handleInvalid='skip'
    )

    df_assembled = assembler.transform(df)

    print(f"Created feature vector with {len(all_feature_cols)} features")
    print(f"Feature columns: {all_feature_cols}")

    return df_assembled, all_feature_cols


def scale_features(df, input_col='features', output_col='scaled_features', method='standard'):
    """
    Scales the feature vector.

    Args:
        df: Input Spark DataFrame with features vector
        input_col: Name of the input features column
        output_col: Name of the output scaled features column
        method: 'standard' for StandardScaler, 'minmax' for MinMaxScaler

    Returns:
        tuple: (transformed DataFrame, scaler model)
    """
    if method == 'standard':
        scaler = StandardScaler(
            inputCol=input_col,
            outputCol=output_col,
            withStd=True,
            withMean=True
        )
    elif method == 'minmax':
        scaler = MinMaxScaler(
            inputCol=input_col,
            outputCol=output_col
        )
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    scaler_model = scaler.fit(df)
    df_scaled = scaler_model.transform(df)

    print(f"Applied {method} scaling to features")

    return df_scaled, scaler_model


def create_feature_pipeline(categorical_cols, numeric_cols, label_col='amount'):
    """
    Creates a complete feature engineering pipeline.

    Args:
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        label_col: Name of the target/label column

    Returns:
        Pipeline: Spark ML Pipeline for feature engineering
    """
    stages = []

    # String indexers for categorical columns
    indexed_cols = []
    for col_name in categorical_cols:
        indexed_col = col_name + '_indexed'
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=indexed_col,
            handleInvalid='keep'
        )
        stages.append(indexer)
        indexed_cols.append(indexed_col)

    # Vector assembler
    all_feature_cols = numeric_cols + indexed_cols
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol='features',
        handleInvalid='skip'
    )
    stages.append(assembler)

    # Standard scaler
    scaler = StandardScaler(
        inputCol='features',
        outputCol='scaled_features',
        withStd=True,
        withMean=False  # Set to False for sparse vectors
    )
    stages.append(scaler)

    pipeline = Pipeline(stages=stages)

    return pipeline, all_feature_cols


def train_test_split(df, train_ratio=0.8, seed=42):
    """
    Splits the DataFrame into training and test sets.

    Args:
        df: Input Spark DataFrame
        train_ratio: Proportion of data for training (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = df.randomSplit([train_ratio, 1 - train_ratio], seed=seed)

    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count

    print(f"Train/Test Split (seed={seed}):")
    print(f"  Training set: {train_count:,} ({train_count/total*100:.1f}%)")
    print(f"  Test set: {test_count:,} ({test_count/total*100:.1f}%)")

    return train_df, test_df


def engineer_features(df, date_col, categorical_cols, numeric_cols, label_col):
    """
    Main feature engineering function that applies all transformations.

    Args:
        df: Cleaned input DataFrame
        date_col: Name of the date column
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        label_col: Name of the target column

    Returns:
        tuple: (feature_df, feature_names)
    """
    print("=" * 60)
    print("STARTING FEATURE ENGINEERING")
    print("=" * 60)

    # Step 1: Extract temporal features
    print("\n[Step 1] Extracting temporal features...")
    df = extract_temporal_features(df, date_col)

    # Add temporal features to numeric columns
    temporal_cols = ['trans_year', 'trans_month', 'trans_quarter', 'trans_dayofweek']
    all_numeric_cols = numeric_cols + temporal_cols

    # Step 2: Encode categorical columns
    print("\n[Step 2] Encoding categorical columns...")
    df, indexed_cols, _ = encode_categorical_columns(df, categorical_cols)

    # Step 3: Create feature vector
    print("\n[Step 3] Creating feature vector...")
    df, feature_names = create_feature_vector(df, all_numeric_cols, indexed_cols)

    # Step 4: Rename label column to 'label' for MLlib compatibility
    df = df.withColumn('label', col(label_col).cast('double'))

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"Total features: {len(feature_names)}")
    print("=" * 60)

    return df, feature_names

"""
Model Training Script for Web Application (PySpark Version)

This script:
1. Loads the Dubai real estate transactions dataset using PySpark
2. Cleans and preprocesses the data (same steps as in the notebook)
3. Trains the custom BaggingRegressor from scratch
4. Saves the model and feature pipeline for the web app

Run this script once to generate the model files needed by the Streamlit app.
"""

import sys
import os
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper, trim, year, month, quarter, dayofweek, to_date, mean
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

from bagging_ensemble import BaggingRegressor, save_bagging_model
from evaluation import calculate_metrics

# Paths
DATA_PATH = Path(__file__).parent.parent / 'data' / 'Transactions.csv'
OUTPUT_DIR = Path(__file__).parent
MODEL_DIR = OUTPUT_DIR / 'spark_model'


def create_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName("bigboyz-web-training") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()


def load_and_clean_data(spark):
    """Load and clean the dataset (same steps as notebook)."""
    print("Loading dataset...")
    df = spark.read.csv(str(DATA_PATH), header=True, inferSchema=True)
    initial_count = df.count()
    print(f"  Loaded {initial_count:,} rows")

    # Filter to sales transactions only
    print("Filtering to sales transactions...")
    df = df.filter(upper(trim(col('trans_group_en'))) == 'SALES')
    print(f"  {df.count():,} sales transactions")

    # Remove duplicates
    print("Removing duplicates...")
    df = df.dropDuplicates()
    print(f"  {df.count():,} rows after deduplication")

    # Handle price outliers using IQR method (factor=3.0)
    print("Handling price outliers...")
    quantiles = df.approxQuantile('actual_worth', [0.25, 0.75], 0.01)
    q1, q3 = quantiles[0], quantiles[1]
    iqr = q3 - q1
    lower_bound = max(0, q1 - 3.0 * iqr)
    upper_bound = q3 + 3.0 * iqr
    df = df.filter((col('actual_worth') >= lower_bound) & (col('actual_worth') <= upper_bound))
    print(f"  Price bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")
    print(f"  {df.count():,} rows after outlier removal")

    return df


def extract_feature_options(df):
    """Extract unique values for categorical features (for dropdowns)."""
    print("\nExtracting feature options for dropdowns...")

    categorical_cols = [
        'property_type_en',
        'property_usage_en',
        'area_name_en',
        'nearest_metro_en',
        'nearest_mall_en'
    ]

    options = {}
    for col_name in categorical_cols:
        unique_vals = df.select(col_name).distinct().collect()
        unique_vals = sorted([row[0] for row in unique_vals if row[0] is not None])
        options[col_name] = unique_vals
        print(f"  {col_name}: {len(unique_vals)} unique values")

    # Save to JSON
    output_path = OUTPUT_DIR / 'feature_options.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(options, f, indent=2, ensure_ascii=False)
    print(f"\nSaved feature options to {output_path}")

    return options


def engineer_features(df):
    """Apply feature engineering (same steps as notebook)."""
    print("\nEngineering features...")

    # Parse date and extract temporal features
    df = df.withColumn('instance_date_parsed', to_date(col('instance_date'), 'dd-MM-yyyy'))
    df = df.withColumn('trans_year', year(col('instance_date_parsed')))
    df = df.withColumn('trans_month', month(col('instance_date_parsed')))
    df = df.withColumn('trans_quarter', quarter(col('instance_date_parsed')))
    df = df.withColumn('trans_dayofweek', dayofweek(col('instance_date_parsed')))

    # Fill missing numeric values
    mean_area = df.select(mean('procedure_area')).collect()[0][0]
    df = df.fillna({'procedure_area': mean_area, 'has_parking': 0})

    # Fill missing categorical values
    categorical_cols = ['property_type_en', 'property_usage_en', 'area_name_en',
                       'nearest_metro_en', 'nearest_mall_en']
    for col_name in categorical_cols:
        df = df.fillna({col_name: 'Unknown'})

    # Drop rows with missing target or temporal features
    df = df.filter(col('actual_worth').isNotNull())
    df = df.filter(col('trans_year').isNotNull())

    # Add label column
    df = df.withColumn('label', col('actual_worth').cast('double'))

    print(f"  {df.count():,} rows after feature engineering")
    return df


def create_feature_pipeline(categorical_cols):
    """Create the feature engineering pipeline."""
    stages = []
    indexed_cols = []

    # StringIndexers for categorical columns
    for col_name in categorical_cols:
        indexed_col = col_name + '_indexed'
        indexer = StringIndexer(
            inputCol=col_name,
            outputCol=indexed_col,
            handleInvalid='keep'
        )
        stages.append(indexer)
        indexed_cols.append(indexed_col)

    # Numeric columns
    numeric_cols = ['procedure_area', 'has_parking', 'trans_year', 'trans_month',
                   'trans_quarter', 'trans_dayofweek']

    # VectorAssembler
    all_feature_cols = numeric_cols + indexed_cols
    assembler = VectorAssembler(
        inputCols=all_feature_cols,
        outputCol='features',
        handleInvalid='skip'
    )
    stages.append(assembler)

    pipeline = Pipeline(stages=stages)
    return pipeline, all_feature_cols


def train_model(df):
    """Train the custom BaggingRegressor."""
    print("\nCreating feature pipeline...")
    categorical_cols = ['property_type_en', 'property_usage_en', 'area_name_en',
                       'nearest_metro_en', 'nearest_mall_en']

    pipeline, feature_cols = create_feature_pipeline(categorical_cols)
    print(f"  Feature columns: {feature_cols}")

    # Fit the feature pipeline and transform data
    print("\nFitting feature pipeline...")
    pipeline_model = pipeline.fit(df)
    df_features = pipeline_model.transform(df)

    # Save the pipeline model
    pipeline_path = str(MODEL_DIR / 'feature_pipeline')
    pipeline_model.write().overwrite().save(pipeline_path)
    print(f"  Saved feature pipeline to {pipeline_path}")

    # Split data
    print("\nSplitting data (80/20)...")
    train_df, test_df = df_features.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  Training set: {train_count:,} samples")
    print(f"  Test set: {test_count:,} samples")

    # Train the custom BaggingRegressor
    print("\nTraining Custom BaggingRegressor (FROM SCRATCH)...")
    model = BaggingRegressor(
        n_estimators=10,
        max_depth=10,
        max_bins=256,
        seed=42,
        features_col='features',
        label_col='label'
    )
    model.fit(train_df, verbose=True)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = model.predict(test_df)
    metrics = calculate_metrics(predictions)
    print(f"  RMSE: {metrics['rmse']:,.2f}")
    print(f"  MAE:  {metrics['mae']:,.2f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    # Save the model
    model_path = str(MODEL_DIR / 'bagging_model')
    save_bagging_model(model, model_path)

    # Save model config
    config = {
        'feature_cols': feature_cols,
        'categorical_cols': categorical_cols,
        'numeric_cols': ['procedure_area', 'has_parking', 'trans_year', 'trans_month',
                        'trans_quarter', 'trans_dayofweek'],
        'target_col': 'actual_worth',
        'model_type': 'Custom BaggingRegressor (FROM SCRATCH)',
        'n_estimators': 10,
        'max_depth': 10,
        'test_metrics': {
            'rmse': float(metrics['rmse']),
            'mae': float(metrics['mae']),
            'r2': float(metrics['r2'])
        }
    }
    config_path = OUTPUT_DIR / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nSaved model config to {config_path}")

    train_df.unpersist()
    return model, pipeline_model


def main():
    print("=" * 60)
    print("TRAINING CUSTOM BAGGING MODEL FOR WEB APPLICATION")
    print("=" * 60)

    # Create Spark session
    spark = create_spark_session()
    print(f"Spark version: {spark.version}")

    try:
        # Load and clean data
        df = load_and_clean_data(spark)

        # Extract feature options for dropdowns
        extract_feature_options(df)

        # Engineer features
        df = engineer_features(df)

        # Train model
        model, pipeline_model = train_model(df)

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - web/spark_model/bagging_model/ (trained custom model)")
        print("  - web/spark_model/feature_pipeline/ (feature transformers)")
        print("  - web/feature_options.json (dropdown values)")
        print("  - web/model_config.json (model configuration)")
        print("\nYou can now run the Streamlit app:")
        print("  cd web && streamlit run app.py")

    finally:
        spark.stop()
        print("\nSpark session stopped.")


if __name__ == '__main__':
    main()

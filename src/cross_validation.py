"""
Manual K-Fold Cross-Validation Module (FROM SCRATCH)

IMPORTANT: This module implements 10-fold cross-validation WITHOUT using
Spark MLlib's CrossValidator or TrainValidationSplit classes.

This is a requirement of the CSCI316 assignment.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, rand, floor, lit, monotonically_increasing_id
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np


def create_folds(df, k=10, seed=42):
    """
    Assigns each row to one of k folds randomly.

    This function manually implements fold assignment WITHOUT using
    any built-in cross-validation utilities.

    Args:
        df: Input Spark DataFrame with 'features' and 'label' columns
        k: Number of folds (default 10)
        seed: Random seed for reproducibility

    Returns:
        DataFrame: DataFrame with additional 'fold' column (values 0 to k-1)
    """
    # Add a random number column and assign fold based on it
    df_with_fold = df.withColumn(
        'fold',
        (floor(rand(seed) * k)).cast('int')
    )

    # Verify fold distribution
    print(f"Fold distribution (k={k}):")
    fold_counts = df_with_fold.groupBy('fold').count().orderBy('fold').collect()
    for row in fold_counts:
        print(f"  Fold {row['fold']}: {row['count']:,} samples")

    return df_with_fold


def get_fold_split(df, fold_num, k=10):
    """
    Returns training and validation DataFrames for a specific fold.

    For fold i:
    - Validation set: rows where fold == i
    - Training set: rows where fold != i

    Args:
        df: DataFrame with 'fold' column
        fold_num: Which fold to use as validation (0 to k-1)
        k: Total number of folds

    Returns:
        tuple: (train_df, val_df)
    """
    if fold_num < 0 or fold_num >= k:
        raise ValueError(f"fold_num must be between 0 and {k-1}")

    val_df = df.filter(col('fold') == fold_num)
    train_df = df.filter(col('fold') != fold_num)

    return train_df, val_df


def calculate_metrics(predictions_df, label_col='label', prediction_col='prediction'):
    """
    Calculates regression metrics for a predictions DataFrame.

    Args:
        predictions_df: DataFrame with actual labels and predictions
        label_col: Name of the label column
        prediction_col: Name of the prediction column

    Returns:
        dict: Dictionary containing RMSE, MAE, and R² metrics
    """
    # RMSE - Root Mean Square Error
    rmse_evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='rmse'
    )
    rmse = rmse_evaluator.evaluate(predictions_df)

    # MAE - Mean Absolute Error
    mae_evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='mae'
    )
    mae = mae_evaluator.evaluate(predictions_df)

    # R² - Coefficient of Determination
    r2_evaluator = RegressionEvaluator(
        labelCol=label_col,
        predictionCol=prediction_col,
        metricName='r2'
    )
    r2 = r2_evaluator.evaluate(predictions_df)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def manual_cross_validate(df, model_builder_fn, k=10, seed=42, verbose=True):
    """
    Performs k-fold cross-validation manually WITHOUT using CrossValidator.

    This is the main function that implements manual cross-validation
    as required by the CSCI316 assignment specification.

    Algorithm:
    1. Assign each row to one of k folds randomly
    2. For each fold i from 0 to k-1:
       a. Use fold i as validation set
       b. Use all other folds as training set
       c. Train a fresh model on training set
       d. Generate predictions on validation set
       e. Calculate and store metrics
    3. Return mean and std of all metrics across folds

    Args:
        df: DataFrame with 'features' and 'label' columns
        model_builder_fn: Function that returns a fresh (untrained) model instance
                         Example: lambda: DecisionTreeRegressor(featuresCol='features', labelCol='label')
        k: Number of folds (default 10)
        seed: Random seed for reproducibility
        verbose: If True, print progress information

    Returns:
        dict: Dictionary containing:
            - 'rmse_mean', 'rmse_std': Mean and std of RMSE across folds
            - 'mae_mean', 'mae_std': Mean and std of MAE across folds
            - 'r2_mean', 'r2_std': Mean and std of R² across folds
            - 'fold_metrics': List of metrics for each fold
    """
    if verbose:
        print("=" * 60)
        print(f"MANUAL {k}-FOLD CROSS-VALIDATION")
        print("=" * 60)

    # Step 1: Assign folds
    if verbose:
        print("\n[Step 1] Assigning folds...")
    df_folded = create_folds(df, k=k, seed=seed)

    # Cache the folded DataFrame for efficiency
    df_folded.cache()

    # Step 2: Iterate through each fold
    fold_metrics = []

    for fold_num in range(k):
        if verbose:
            print(f"\n[Fold {fold_num + 1}/{k}] Training and evaluating...")

        # Get train/validation split for this fold
        train_df, val_df = get_fold_split(df_folded, fold_num, k)

        # Create a fresh model instance
        model = model_builder_fn()

        # Train the model on training data
        trained_model = model.fit(train_df)

        # Generate predictions on validation data
        predictions = trained_model.transform(val_df)

        # Calculate metrics
        metrics = calculate_metrics(predictions)
        fold_metrics.append(metrics)

        if verbose:
            print(f"       RMSE: {metrics['rmse']:,.2f}")
            print(f"       MAE:  {metrics['mae']:,.2f}")
            print(f"       R²:   {metrics['r2']:.4f}")

    # Step 3: Aggregate metrics across all folds
    rmse_values = [m['rmse'] for m in fold_metrics]
    mae_values = [m['mae'] for m in fold_metrics]
    r2_values = [m['r2'] for m in fold_metrics]

    results = {
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'mae_mean': np.mean(mae_values),
        'mae_std': np.std(mae_values),
        'r2_mean': np.mean(r2_values),
        'r2_std': np.std(r2_values),
        'fold_metrics': fold_metrics
    }

    # Unpersist cached DataFrame
    df_folded.unpersist()

    if verbose:
        print("\n" + "=" * 60)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 60)
        print(f"RMSE: {results['rmse_mean']:,.2f} (+/- {results['rmse_std']:,.2f})")
        print(f"MAE:  {results['mae_mean']:,.2f} (+/- {results['mae_std']:,.2f})")
        print(f"R²:   {results['r2_mean']:.4f} (+/- {results['r2_std']:.4f})")
        print("=" * 60)

    return results


def compare_models_cv(df, model_configs, k=10, seed=42):
    """
    Compares multiple models using manual cross-validation.

    Args:
        df: DataFrame with 'features' and 'label' columns
        model_configs: Dictionary mapping model names to builder functions
                      Example: {'DecisionTree': lambda: DecisionTreeRegressor(...)}
        k: Number of folds
        seed: Random seed

    Returns:
        dict: Dictionary mapping model names to their CV results
    """
    print("=" * 60)
    print(f"COMPARING {len(model_configs)} MODELS WITH {k}-FOLD CV")
    print("=" * 60)

    results = {}

    for model_name, model_builder in model_configs.items():
        print(f"\n>>> Evaluating: {model_name}")
        cv_results = manual_cross_validate(
            df,
            model_builder,
            k=k,
            seed=seed,
            verbose=False
        )
        results[model_name] = cv_results

        print(f"    RMSE: {cv_results['rmse_mean']:,.2f} (+/- {cv_results['rmse_std']:,.2f})")
        print(f"    MAE:  {cv_results['mae_mean']:,.2f} (+/- {cv_results['mae_std']:,.2f})")
        print(f"    R²:   {cv_results['r2_mean']:.4f} (+/- {cv_results['r2_std']:.4f})")

    return results

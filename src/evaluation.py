"""
Evaluation Module
Provides metrics calculation, model evaluation, and visualization functions.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, abs as spark_abs, pow as spark_pow, avg, sqrt
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def calculate_metrics(predictions_df, label_col='label', prediction_col='prediction'):
    """
    Calculates comprehensive regression metrics.

    Args:
        predictions_df: DataFrame with actual labels and predictions
        label_col: Name of the label column
        prediction_col: Name of the prediction column

    Returns:
        dict: Dictionary containing RMSE, MAE, R², MAPE metrics
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

    # MAPE - Mean Absolute Percentage Error (calculated manually)
    # Filter out zero labels to avoid division by zero
    mape_df = predictions_df.filter(col(label_col) != 0)
    mape = mape_df.select(
        avg(spark_abs((col(label_col) - col(prediction_col)) / col(label_col)) * 100)
    ).collect()[0][0]

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape if mape else 0.0
    }


def evaluate_model(model, test_df, model_name='Model', verbose=True):
    """
    Evaluates a trained model on test data.

    Args:
        model: Trained model with transform() method
        test_df: Test DataFrame with 'features' and 'label' columns
        model_name: Name of the model for display
        verbose: If True, print evaluation results

    Returns:
        tuple: (predictions_df, metrics_dict)
    """
    # Generate predictions
    predictions = model.transform(test_df)

    # Calculate metrics
    metrics = calculate_metrics(predictions)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"EVALUATION RESULTS: {model_name}")
        print(f"{'=' * 50}")
        print(f"  RMSE: {metrics['rmse']:,.2f}")
        print(f"  MAE:  {metrics['mae']:,.2f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"{'=' * 50}")

    return predictions, metrics


def create_comparison_table(results_dict):
    """
    Creates a comparison table from model results.

    Args:
        results_dict: Dictionary mapping model names to their metrics
                     Can be CV results or test evaluation results

    Returns:
        pandas.DataFrame: Comparison table
    """
    rows = []

    for model_name, metrics in results_dict.items():
        if 'rmse_mean' in metrics:
            # CV results format
            row = {
                'Model': model_name,
                'RMSE': f"{metrics['rmse_mean']:,.0f} ± {metrics['rmse_std']:,.0f}",
                'MAE': f"{metrics['mae_mean']:,.0f} ± {metrics['mae_std']:,.0f}",
                'R²': f"{metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}",
                'RMSE_val': metrics['rmse_mean'],
                'R2_val': metrics['r2_mean']
            }
        else:
            # Direct metrics format
            row = {
                'Model': model_name,
                'RMSE': f"{metrics['rmse']:,.0f}",
                'MAE': f"{metrics['mae']:,.0f}",
                'R²': f"{metrics['r2']:.4f}",
                'RMSE_val': metrics['rmse'],
                'R2_val': metrics['r2']
            }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by R² descending
    df = df.sort_values('R2_val', ascending=False)

    return df[['Model', 'RMSE', 'MAE', 'R²']]


def plot_model_comparison(results_dict, save_path=None):
    """
    Creates a bar chart comparing model performance.

    Args:
        results_dict: Dictionary mapping model names to their metrics
        save_path: If provided, save figure to this path
    """
    models = list(results_dict.keys())

    # Extract metrics
    if 'rmse_mean' in list(results_dict.values())[0]:
        rmse_vals = [results_dict[m]['rmse_mean'] for m in models]
        r2_vals = [results_dict[m]['r2_mean'] for m in models]
    else:
        rmse_vals = [results_dict[m]['rmse'] for m in models]
        r2_vals = [results_dict[m]['r2'] for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE comparison
    colors = sns.color_palette("husl", len(models))
    bars1 = axes[0].bar(models, rmse_vals, color=colors)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('RMSE (AED)')
    axes[0].set_title('Model Comparison - RMSE (Lower is Better)')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, rmse_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

    # R² comparison
    bars2 = axes[1].bar(models, r2_vals, color=colors)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Model Comparison - R² (Higher is Better)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, 1)
    for bar, val in zip(bars2, r2_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_predictions_vs_actual(predictions_df, sample_size=10000, save_path=None):
    """
    Creates a scatter plot of predicted vs actual values.

    Args:
        predictions_df: DataFrame with 'label' and 'prediction' columns
        sample_size: Number of points to plot (for performance)
        save_path: If provided, save figure to this path
    """
    # Sample data for plotting
    sample_df = predictions_df.select('label', 'prediction').sample(
        False, min(1.0, sample_size / predictions_df.count())
    ).toPandas()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(sample_df['label'], sample_df['prediction'],
              alpha=0.5, s=10, c='steelblue')

    # Perfect prediction line
    max_val = max(sample_df['label'].max(), sample_df['prediction'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual Price (AED)')
    ax.set_ylabel('Predicted Price (AED)')
    ax.set_title('Predicted vs Actual Prices')
    ax.legend()

    # Format axis labels
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(6, 6))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_residuals(predictions_df, sample_size=10000, save_path=None):
    """
    Creates residual plots for model diagnostics.

    Args:
        predictions_df: DataFrame with 'label' and 'prediction' columns
        sample_size: Number of points to plot
        save_path: If provided, save figure to this path
    """
    # Calculate residuals and sample
    residuals_df = predictions_df.withColumn(
        'residual', col('prediction') - col('label')
    )

    sample_df = residuals_df.select('label', 'prediction', 'residual').sample(
        False, min(1.0, sample_size / residuals_df.count())
    ).toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(sample_df['prediction'], sample_df['residual'],
                   alpha=0.5, s=10, c='steelblue')
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Price (AED)')
    axes[0].set_ylabel('Residual (AED)')
    axes[0].set_title('Residuals vs Predicted Values')

    # Residual distribution
    axes[1].hist(sample_df['residual'], bins=50, color='steelblue', edgecolor='white')
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].set_xlabel('Residual (AED)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_feature_importance(importance_dict, top_n=15, save_path=None):
    """
    Creates a horizontal bar chart of feature importance.

    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        top_n: Number of top features to display
        save_path: If provided, save figure to this path
    """
    # Sort and get top N
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_importance[:top_n]

    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = sns.color_palette("viridis", len(features))
    bars = ax.barh(features, importances, color=colors)

    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.invert_yaxis()  # Top feature at top

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()


def plot_price_distribution(df, price_col='amount', save_path=None):
    """
    Creates a histogram of price distribution.

    Args:
        df: DataFrame with price column
        price_col: Name of the price column
        save_path: If provided, save figure to this path
    """
    # Sample for plotting
    sample_df = df.select(price_col).sample(False, 0.1).toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original scale
    axes[0].hist(sample_df[price_col], bins=50, color='steelblue', edgecolor='white')
    axes[0].set_xlabel('Price (AED)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Price Distribution')

    # Log scale
    log_prices = np.log1p(sample_df[price_col])
    axes[1].hist(log_prices, bins=50, color='coral', edgecolor='white')
    axes[1].set_xlabel('Log(Price + 1)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Price Distribution (Log Scale)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    plt.show()

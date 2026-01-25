"""
Data Cleaning Module
Handles preprocessing, cleaning, and filtering of raw transaction data.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, isnan, isnull, count, trim, lower, upper,
    to_date, year, month, dayofmonth, quarter, dayofweek,
    regexp_replace, lit, avg, stddev
)
from pyspark.sql.types import DoubleType, IntegerType


def analyze_missing_values(df):
    """
    Analyzes missing values in the DataFrame.

    Args:
        df: Input Spark DataFrame

    Returns:
        DataFrame: Summary of missing values per column
    """
    total_rows = df.count()

    missing_data = []
    for column in df.columns:
        dtype = df.schema[column].dataType.typeName()

        if dtype in ['double', 'float', 'integer', 'long']:
            missing_count = df.filter(col(column).isNull() | isnan(col(column))).count()
        else:
            missing_count = df.filter(col(column).isNull()).count()

        missing_pct = (missing_count / total_rows) * 100
        missing_data.append({
            'column': column,
            'missing_count': missing_count,
            'missing_pct': round(missing_pct, 2),
            'dtype': dtype
        })

    return missing_data


def drop_high_missing_columns(df, threshold=50.0):
    """
    Drops columns with missing value percentage above threshold.

    Args:
        df: Input Spark DataFrame
        threshold: Maximum allowed missing percentage (default 50%)

    Returns:
        DataFrame: DataFrame with high-missing columns removed
    """
    total_rows = df.count()
    columns_to_drop = []

    for column in df.columns:
        dtype = df.schema[column].dataType.typeName()

        if dtype in ['double', 'float', 'integer', 'long']:
            missing_count = df.filter(col(column).isNull() | isnan(col(column))).count()
        else:
            missing_count = df.filter(col(column).isNull()).count()

        missing_pct = (missing_count / total_rows) * 100

        if missing_pct > threshold:
            columns_to_drop.append(column)
            print(f"Dropping column '{column}' ({missing_pct:.1f}% missing)")

    return df.drop(*columns_to_drop)


def filter_transaction_type(df, transaction_type_col, keep_types=None):
    """
    Filters DataFrame to keep only specified transaction types.
    Typically we want to keep only 'Sales' transactions for price prediction.

    Args:
        df: Input Spark DataFrame
        transaction_type_col: Name of the column containing transaction type
        keep_types: List of transaction types to keep (e.g., ['Sales'])

    Returns:
        DataFrame: Filtered DataFrame
    """
    if keep_types is None:
        keep_types = ['Sales']  # Dubai Land Department uses 'Sales' for sale transactions

    # Normalize and filter
    df_filtered = df.filter(
        upper(trim(col(transaction_type_col))).isin([t.upper() for t in keep_types])
    )

    original_count = df.count()
    filtered_count = df_filtered.count()
    print(f"Filtered transactions: {original_count:,} -> {filtered_count:,}")
    print(f"Removed {original_count - filtered_count:,} non-sale transactions")

    return df_filtered


def remove_duplicates(df, subset_cols=None):
    """
    Removes duplicate rows from the DataFrame.

    Args:
        df: Input Spark DataFrame
        subset_cols: List of columns to consider for duplicate detection
                    If None, considers all columns

    Returns:
        DataFrame: DataFrame with duplicates removed
    """
    original_count = df.count()

    if subset_cols:
        df_deduped = df.dropDuplicates(subset_cols)
    else:
        df_deduped = df.dropDuplicates()

    deduped_count = df_deduped.count()
    print(f"Removed {original_count - deduped_count:,} duplicate rows")

    return df_deduped


def handle_price_outliers(df, price_col, method='iqr', factor=1.5):
    """
    Handles outliers in the price column.

    Args:
        df: Input Spark DataFrame
        price_col: Name of the price/amount column
        method: 'iqr' for IQR method, 'percentile' for percentile capping
        factor: IQR multiplier for outlier detection (default 1.5)

    Returns:
        DataFrame: DataFrame with outliers handled
    """
    original_count = df.count()

    if method == 'iqr':
        # Calculate IQR
        quantiles = df.approxQuantile(price_col, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1

        lower_bound = q1 - (factor * iqr)
        upper_bound = q3 + (factor * iqr)

        # Ensure lower bound is not negative for prices
        lower_bound = max(0, lower_bound)

        print(f"Price outlier bounds (IQR method): [{lower_bound:,.0f}, {upper_bound:,.0f}]")

        df_filtered = df.filter(
            (col(price_col) >= lower_bound) &
            (col(price_col) <= upper_bound)
        )

    elif method == 'percentile':
        # Cap at 1st and 99th percentile
        quantiles = df.approxQuantile(price_col, [0.01, 0.99], 0.001)
        lower_bound, upper_bound = quantiles[0], quantiles[1]

        print(f"Price outlier bounds (percentile method): [{lower_bound:,.0f}, {upper_bound:,.0f}]")

        df_filtered = df.filter(
            (col(price_col) >= lower_bound) &
            (col(price_col) <= upper_bound)
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    filtered_count = df_filtered.count()
    print(f"Removed {original_count - filtered_count:,} price outliers")

    return df_filtered


def handle_missing_values(df, numeric_strategy='mean', categorical_strategy='mode'):
    """
    Handles missing values in the DataFrame.

    Args:
        df: Input Spark DataFrame
        numeric_strategy: 'mean', 'median', or 'drop' for numeric columns
        categorical_strategy: 'mode', 'unknown', or 'drop' for categorical columns

    Returns:
        DataFrame: DataFrame with missing values handled
    """
    from pyspark.sql.functions import mean, mode

    result_df = df

    for column in df.columns:
        dtype = df.schema[column].dataType.typeName()

        if dtype in ['double', 'float', 'integer', 'long']:
            # Numeric column
            if numeric_strategy == 'mean':
                mean_val = df.select(mean(col(column))).collect()[0][0]
                if mean_val is not None:
                    result_df = result_df.fillna({column: mean_val})
            elif numeric_strategy == 'drop':
                result_df = result_df.filter(col(column).isNotNull() & ~isnan(col(column)))
        else:
            # Categorical column
            if categorical_strategy == 'unknown':
                result_df = result_df.fillna({column: 'Unknown'})
            elif categorical_strategy == 'drop':
                result_df = result_df.filter(col(column).isNotNull())

    return result_df


def clean_data(df, price_col='amount', transaction_type_col='transaction_type'):
    """
    Main cleaning pipeline that applies all cleaning steps.

    Args:
        df: Raw input DataFrame
        price_col: Name of the price/amount column
        transaction_type_col: Name of the transaction type column

    Returns:
        DataFrame: Cleaned DataFrame ready for feature engineering
    """
    print("=" * 60)
    print("STARTING DATA CLEANING PIPELINE")
    print("=" * 60)

    initial_count = df.count()
    print(f"\nInitial row count: {initial_count:,}")

    # Step 1: Remove duplicates
    print("\n[Step 1] Removing duplicates...")
    df = remove_duplicates(df)

    # Step 2: Filter to sales transactions only
    print("\n[Step 2] Filtering to sales transactions...")
    df = filter_transaction_type(df, transaction_type_col)

    # Step 3: Handle price outliers
    print("\n[Step 3] Handling price outliers...")
    df = handle_price_outliers(df, price_col, method='iqr', factor=3.0)

    # Step 4: Drop columns with too many missing values
    print("\n[Step 4] Dropping high-missing columns...")
    df = drop_high_missing_columns(df, threshold=50.0)

    # Step 5: Handle remaining missing values
    print("\n[Step 5] Handling remaining missing values...")
    df = handle_missing_values(df, numeric_strategy='mean', categorical_strategy='unknown')

    final_count = df.count()
    print("\n" + "=" * 60)
    print(f"CLEANING COMPLETE: {initial_count:,} -> {final_count:,} rows")
    print(f"Retained {(final_count/initial_count)*100:.1f}% of data")
    print("=" * 60)

    return df

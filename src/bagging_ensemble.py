"""
Bagging Ensemble Regressor Module (FROM SCRATCH)

IMPORTANT: This module implements a Bagging ensemble WITHOUT using
Spark MLlib's RandomForestRegressor class.

This is a requirement of the CSCI316 assignment.

The implementation uses DecisionTreeRegressor from MLlib as base learners,
but implements the bagging logic (bootstrap sampling, training multiple
models, and averaging predictions) from scratch.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, avg, lit, monotonically_increasing_id
from pyspark.ml.regression import DecisionTreeRegressor
import numpy as np


def bootstrap_sample(df, fraction=1.0, seed=None):
    """
    Creates a bootstrap sample from the DataFrame (sampling WITH replacement).

    This manually implements bootstrap sampling as required by the assignment.

    Args:
        df: Input Spark DataFrame
        fraction: Fraction of data to sample (default 1.0 = same size as original)
        seed: Random seed for reproducibility

    Returns:
        DataFrame: Bootstrap sample (may contain duplicate rows)
    """
    # Spark's sample() with replacement=True implements bootstrap sampling
    # fraction=1.0 means we sample approximately the same number of rows as original
    bootstrap_df = df.sample(
        withReplacement=True,
        fraction=fraction,
        seed=seed
    )

    return bootstrap_df


class BaggingRegressor:
    """
    Custom Bagging Ensemble Regressor implemented FROM SCRATCH.

    This class implements the bagging algorithm:
    1. Create n bootstrap samples from training data
    2. Train a DecisionTreeRegressor on each bootstrap sample
    3. For predictions, average the predictions from all trees

    This implementation does NOT use RandomForestRegressor from MLlib,
    as per the assignment requirements.

    Attributes:
        n_estimators: Number of base models (trees) in the ensemble
        max_depth: Maximum depth of each decision tree
        min_instances_per_node: Minimum samples required at a leaf node
        seed: Random seed for reproducibility
        models: List of trained DecisionTreeRegressorModel instances
        feature_col: Name of the features column
        label_col: Name of the label column
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=10,
        min_instances_per_node=1,
        max_bins=256,
        seed=42,
        features_col='features',
        label_col='label'
    ):
        """
        Initializes the BaggingRegressor.

        Args:
            n_estimators: Number of trees in the ensemble (default 10)
            max_depth: Maximum depth of each tree (default 10)
            min_instances_per_node: Minimum instances per leaf node (default 1)
            max_bins: Max bins for discretizing continuous features and
                     determining splits for categorical features (default 256)
            seed: Random seed for reproducibility (default 42)
            features_col: Name of features column (default 'features')
            label_col: Name of label column (default 'label')
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_instances_per_node = min_instances_per_node
        self.max_bins = max_bins
        self.seed = seed
        self.features_col = features_col
        self.label_col = label_col
        self.models = []
        self._is_fitted = False

    def fit(self, train_df, verbose=True):
        """
        Trains the bagging ensemble on the training data.

        Algorithm:
        1. For each estimator i from 0 to n_estimators-1:
           a. Create a bootstrap sample from train_df
           b. Create a new DecisionTreeRegressor
           c. Train the tree on the bootstrap sample
           d. Store the trained model

        Args:
            train_df: Training DataFrame with 'features' and 'label' columns
            verbose: If True, print progress information

        Returns:
            self: The fitted BaggingRegressor instance
        """
        if verbose:
            print("=" * 60)
            print(f"TRAINING BAGGING ENSEMBLE ({self.n_estimators} estimators)")
            print("=" * 60)
            print(f"  Max depth: {self.max_depth}")
            print(f"  Max bins: {self.max_bins}")
            print(f"  Min instances per node: {self.min_instances_per_node}")
            print(f"  Random seed: {self.seed}")

        self.models = []

        for i in range(self.n_estimators):
            if verbose:
                print(f"\n[Estimator {i + 1}/{self.n_estimators}] ", end="")

            # Step 1: Create bootstrap sample
            # Use different seed for each tree
            tree_seed = self.seed + i if self.seed is not None else None
            bootstrap_df = bootstrap_sample(train_df, fraction=1.0, seed=tree_seed)

            if verbose:
                print("Bootstrap sampling... ", end="")

            # Step 2: Create and configure a DecisionTreeRegressor
            tree = DecisionTreeRegressor(
                featuresCol=self.features_col,
                labelCol=self.label_col,
                predictionCol=f'prediction_{i}',  # Unique prediction column
                maxDepth=self.max_depth,
                minInstancesPerNode=self.min_instances_per_node,
                maxBins=self.max_bins,
                seed=tree_seed
            )

            # Step 3: Train the tree on the bootstrap sample
            if verbose:
                print("Training... ", end="")

            trained_tree = tree.fit(bootstrap_df)

            # Step 4: Store the trained model
            self.models.append(trained_tree)

            if verbose:
                print("Done!")

        self._is_fitted = True

        if verbose:
            print("\n" + "=" * 60)
            print(f"BAGGING ENSEMBLE TRAINING COMPLETE")
            print(f"Trained {len(self.models)} decision trees")
            print("=" * 60)

        return self

    def predict(self, test_df):
        """
        Generates predictions by averaging predictions from all base models.

        Algorithm:
        1. For each trained tree, generate predictions
        2. Average all predictions to get final prediction

        Args:
            test_df: Test DataFrame with 'features' column

        Returns:
            DataFrame: DataFrame with 'prediction' column containing averaged predictions
        """
        if not self._is_fitted:
            raise RuntimeError("BaggingRegressor must be fitted before calling predict()")

        # Start with the test DataFrame
        result_df = test_df

        # Generate predictions from each tree
        prediction_cols = []

        for i, model in enumerate(self.models):
            pred_col = f'prediction_{i}'
            result_df = model.transform(result_df)
            prediction_cols.append(pred_col)

        # Average all predictions
        # Create the averaging expression
        avg_expr = sum([col(pc) for pc in prediction_cols]) / len(prediction_cols)
        result_df = result_df.withColumn('prediction', avg_expr)

        # Drop individual prediction columns (keep only the averaged 'prediction')
        result_df = result_df.drop(*prediction_cols)

        return result_df

    def transform(self, test_df):
        """
        Alias for predict() to maintain compatibility with Spark MLlib interface.

        This allows BaggingRegressor to be used with manual_cross_validate()
        which expects models to have a transform() method.

        Args:
            test_df: Test DataFrame with 'features' column

        Returns:
            DataFrame: DataFrame with 'prediction' column
        """
        return self.predict(test_df)

    def get_feature_importance(self, feature_names=None):
        """
        Aggregates feature importance across all base models.

        Args:
            feature_names: Optional list of feature names

        Returns:
            dict: Dictionary mapping feature index (or name) to importance score
        """
        if not self._is_fitted:
            raise RuntimeError("BaggingRegressor must be fitted before getting feature importance")

        # Get feature importances from each tree and average them
        n_features = len(self.models[0].featureImportances)
        total_importance = np.zeros(n_features)

        for model in self.models:
            importance = model.featureImportances.toArray()
            total_importance += importance

        avg_importance = total_importance / len(self.models)

        # Create result dictionary
        if feature_names is not None and len(feature_names) == n_features:
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, avg_importance)}
        else:
            importance_dict = {f'feature_{i}': float(imp) for i, imp in enumerate(avg_importance)}

        # Sort by importance (descending)
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        return importance_dict

    def get_params(self):
        """Returns the parameters of the BaggingRegressor."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_bins': self.max_bins,
            'min_instances_per_node': self.min_instances_per_node,
            'seed': self.seed,
            'features_col': self.features_col,
            'label_col': self.label_col
        }

    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"BaggingRegressor(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, {status})"
        )


def save_bagging_model(model, path):
    """
    Saves a trained BaggingRegressor to disk.

    Each tree model is saved to a separate directory.
    Model parameters are saved to a JSON file.

    Args:
        model: Trained BaggingRegressor instance
        path: Directory path to save the model
    """
    import os
    import json

    if not model._is_fitted:
        raise RuntimeError("Cannot save unfitted model")

    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Save each tree model
    for i, tree_model in enumerate(model.models):
        tree_path = os.path.join(path, f"tree_{i}")
        tree_model.write().overwrite().save(tree_path)

    # Save model parameters
    params = model.get_params()
    params['n_trees_saved'] = len(model.models)
    params_path = os.path.join(path, "params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"Saved BaggingRegressor with {len(model.models)} trees to {path}")


def load_bagging_model(path):
    """
    Loads a trained BaggingRegressor from disk.

    Args:
        path: Directory path where the model was saved

    Returns:
        BaggingRegressor: Loaded model ready for predictions
    """
    import os
    import json
    from pyspark.ml.regression import DecisionTreeRegressionModel

    # Load parameters
    params_path = os.path.join(path, "params.json")
    with open(params_path, 'r') as f:
        params = json.load(f)

    # Create BaggingRegressor with saved parameters
    model = BaggingRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_bins=params['max_bins'],
        min_instances_per_node=params['min_instances_per_node'],
        seed=params['seed'],
        features_col=params['features_col'],
        label_col=params['label_col']
    )

    # Load each tree model
    n_trees = params['n_trees_saved']
    model.models = []
    for i in range(n_trees):
        tree_path = os.path.join(path, f"tree_{i}")
        tree_model = DecisionTreeRegressionModel.load(tree_path)
        model.models.append(tree_model)

    model._is_fitted = True
    print(f"Loaded BaggingRegressor with {len(model.models)} trees from {path}")

    return model


def create_bagging_model_builder(n_estimators=10, max_depth=10, max_bins=256, seed=42):
    """
    Factory function that creates a BaggingRegressor builder function.

    This is useful for use with manual_cross_validate() which expects
    a function that returns a fresh model instance.

    Args:
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        max_bins: Max bins for categorical features (must be >= max unique values)
        seed: Random seed

    Returns:
        function: A function that returns a new BaggingRegressor instance
    """
    def builder():
        return BaggingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_bins=max_bins,
            seed=seed
        )
    return builder

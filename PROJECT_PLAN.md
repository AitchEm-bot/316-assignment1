# PROJECT_PLAN.md - Detailed Implementation Guide

## Project: bigboyz - Dubai Real Estate Price Prediction

---

## Table of Contents

1. [Project Directory Structure](#project-directory-structure)
2. [Architecture Overview](#architecture-overview)
3. [Implementation Phases](#implementation-phases)
4. [Detailed Checklists](#detailed-checklists)
5. [Timeline](#timeline)
6. [Technical Specifications](#technical-specifications)
7. [Evaluation Strategy](#evaluation-strategy)
8. [Deliverables](#deliverables)

---

## Project Directory Structure

```
bigboyz-dubai-real-estate/
│
├── docker/
│   ├── Dockerfile                # Container definition
│   └── docker-compose.yml        # Multi-container orchestration
│
├── data/
│   ├── .gitkeep                  # Placeholder for git
│   └── transactions.csv          # Raw data (gitignored, ~1.5M rows)
│
├── notebooks/
│   └── bigboyz.ipynb             # MAIN DELIVERABLE NOTEBOOK
│
├── src/
│   ├── __init__.py               # Package initializer
│   ├── data_ingestion.py         # Spark data loading utilities
│   ├── data_cleaning.py          # Preprocessing & cleaning functions
│   ├── feature_engineering.py    # Feature transformation pipeline
│   ├── cross_validation.py       # Manual 10-fold CV (FROM SCRATCH)
│   ├── bagging_ensemble.py       # Bagging implementation (FROM SCRATCH)
│   └── evaluation.py             # Metrics & visualization functions
│
├── outputs/
│   ├── figures/                  # Generated charts/plots for report
│   │   ├── price_distribution.png
│   │   ├── feature_importance.png
│   │   ├── model_comparison.png
│   │   └── cv_results.png
│   └── models/                   # Saved model artifacts (optional)
│
├── report/
│   ├── bigboyz_report.pdf        # Final report
│   └── figures/                  # High-res figures for report
│
├── presentation/
│   ├── bigboyz_slides.pptx       # Presentation slides
│   └── video_link.txt            # YouTube video URL
│
├── CLAUDE.md                     # Project context for AI assistance
├── PROJECT_PLAN.md               # This file
├── README.md                     # Setup & run instructions
├── requirements.txt              # Python dependencies
└── .gitignore                    # Git ignore rules
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DOCKER CONTAINER                               │
│                     (jupyter/pyspark-notebook)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   PHASE 1   │    │   PHASE 2   │    │   PHASE 3   │                 │
│  │   INGEST    │───▶│    CLEAN    │───▶│   FEATURE   │                 │
│  │             │    │             │    │ ENGINEERING │                 │
│  │ • Load CSV  │    │ • Nulls     │    │ • Encoding  │                 │
│  │ • Schema    │    │ • Outliers  │    │ • Scaling   │                 │
│  │ • Explore   │    │ • Duplicates│    │ • Vectors   │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│                                               │                         │
│                                               ▼                         │
│                           ┌───────────────────────────────┐             │
│                           │      80/20 TRAIN SPLIT        │             │
│                           └───────────────────────────────┘             │
│                              │                    │                     │
│                    ┌─────────┘                    └──────────┐          │
│                    ▼                                        ▼          │
│  ┌──────────────────────────────────┐    ┌────────────────────────┐    │
│  │         TRAINING (80%)           │    │   HOLDOUT TEST (20%)   │    │
│  │                                  │    │                        │    │
│  │  ┌────────────────────────────┐  │    │   • Untouched until    │    │
│  │  │   PHASE 4: 10-FOLD CV      │  │    │     final evaluation   │    │
│  │  │      (FROM SCRATCH)        │  │    │                        │    │
│  │  │                            │  │    │   • True unseen data   │    │
│  │  │   Fold 1: Train/Val        │  │    │     performance        │    │
│  │  │   Fold 2: Train/Val        │  │    │                        │    │
│  │  │   ...                      │  │    └────────────────────────┘    │
│  │  │   Fold 10: Train/Val       │  │                                  │
│  │  └────────────────────────────┘  │                                  │
│  │                                  │                                  │
│  │  ┌────────────────────────────┐  │                                  │
│  │  │   PHASE 5: MODELS          │  │                                  │
│  │  │                            │  │                                  │
│  │  │   Baselines (MLlib):       │  │                                  │
│  │  │   • Linear Regression      │  │                                  │
│  │  │   • Decision Tree          │  │                                  │
│  │  │   • Random Forest          │  │                                  │
│  │  │                            │  │                                  │
│  │  │   Custom (FROM SCRATCH):   │  │                                  │
│  │  │   • Bagging Ensemble       │  │                                  │
│  │  └────────────────────────────┘  │                                  │
│  │                                  │                                  │
│  └──────────────────────────────────┘                                  │
│                    │                                                    │
│                    ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    PHASE 6: EVALUATION                            │  │
│  │                                                                   │  │
│  │   • Calculate RMSE, MAE, R² for all models via CV                │  │
│  │   • Train final models on full 80% training set                  │  │
│  │   • Evaluate on 20% holdout test set                             │  │
│  │   • Generate comparison charts & feature importance              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Project Setup & Data Exploration
**Duration:** Day 1-3  
**Goal:** Set up environment and understand the data

#### Tasks:
1. Create project directory structure
2. Write Dockerfile with PySpark environment
3. Create docker-compose.yml
4. Download dataset from Dubai Pulse
5. Initial data exploration in notebook
6. Document dataset characteristics

#### Expected Outputs:
- Working Docker environment
- `bigboyz.ipynb` with exploration cells
- Understanding of data schema, size, missing values

---

### Phase 2: Data Cleaning Pipeline
**Duration:** Day 4-6  
**Goal:** Clean and preprocess raw data using PySpark

#### Tasks:
1. Load CSV into Spark DataFrame with proper schema
2. Analyze missing values per column
3. Handle missing values (drop rows or impute)
4. Remove duplicate transactions
5. Filter to relevant transaction types (sales only)
6. Handle price outliers (remove extremes or cap)
7. Validate data types and fix inconsistencies

#### Expected Outputs:
- `src/data_ingestion.py` with load functions
- `src/data_cleaning.py` with cleaning pipeline
- Clean DataFrame ready for feature engineering

---

### Phase 3: Feature Engineering
**Duration:** Day 7-9  
**Goal:** Transform raw features into ML-ready format

#### Tasks:
1. Identify categorical columns (area, property type, etc.)
2. Encode categorical variables using StringIndexer + OneHotEncoder
3. Extract temporal features from transaction date:
   - Year
   - Month
   - Quarter
   - Day of week
4. Scale numerical features (size, price per sqft, etc.)
5. Create VectorAssembler pipeline for final feature vector
6. Perform 80/20 train/holdout split

#### Expected Outputs:
- `src/feature_engineering.py` with transformation functions
- Feature-engineered DataFrame with `features` and `label` columns
- Train set (~1.2M rows) and holdout test set (~300K rows)

---

### Phase 4: Manual Cross-Validation (FROM SCRATCH)
**Duration:** Day 10-12  
**Goal:** Implement 10-fold CV without using library functions

#### Tasks:
1. Implement `create_folds(df, k=10)` function:
   - Add random fold assignment column
   - Ensure roughly equal fold sizes
2. Implement `manual_cross_validate(df, model_class, k=10)` function:
   - Loop through each fold
   - Create train/validation splits
   - Fit model on training folds
   - Evaluate on validation fold
   - Collect metrics
3. Implement metric aggregation (mean ± std)
4. Test with simple Decision Tree to verify correctness

#### Key Code Structure:
```python
# src/cross_validation.py

def create_folds(df, k=10, seed=42):
    """
    Assigns each row to one of k folds randomly.
    Returns DataFrame with 'fold' column (0 to k-1).
    """
    pass

def get_fold_split(df, fold_num, k=10):
    """
    Returns (train_df, val_df) for given fold number.
    val_df = rows where fold == fold_num
    train_df = rows where fold != fold_num
    """
    pass

def manual_cross_validate(df, model_builder_fn, k=10):
    """
    Performs k-fold cross-validation manually.
    
    Args:
        df: DataFrame with 'features' and 'label' columns
        model_builder_fn: Function that returns a fresh model instance
        k: Number of folds
    
    Returns:
        dict with mean and std of RMSE, MAE, R²
    """
    pass
```

#### Expected Outputs:
- `src/cross_validation.py` with manual CV implementation
- Verified working CV on a test model

---

### Phase 5: Bagging Ensemble (FROM SCRATCH)
**Duration:** Day 13-15  
**Goal:** Implement bagging regressor without using RandomForest

#### Tasks:
1. Implement `bootstrap_sample(df, fraction=1.0, seed=None)`:
   - Sample with replacement
   - Same size as original (or configurable)
2. Implement `BaggingRegressor` class:
   - `__init__(n_estimators, base_model_params)`
   - `fit(train_df)` - trains N trees on bootstrap samples
   - `predict(test_df)` - averages predictions from all trees
3. Add feature importance aggregation (optional but nice)
4. Test with manual CV

#### Key Code Structure:
```python
# src/bagging_ensemble.py

from pyspark.ml.regression import DecisionTreeRegressor

def bootstrap_sample(df, fraction=1.0, seed=None):
    """
    Creates a bootstrap sample (sampling WITH replacement).
    """
    pass

class BaggingRegressor:
    """
    Custom Bagging Ensemble Regressor.
    Uses DecisionTreeRegressor as base learners.
    """
    
    def __init__(self, n_estimators=10, max_depth=10, seed=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.models = []
    
    def fit(self, train_df):
        """
        Trains n_estimators Decision Trees on bootstrap samples.
        """
        pass
    
    def predict(self, test_df):
        """
        Generates predictions by averaging all base model predictions.
        Returns DataFrame with 'prediction' column.
        """
        pass
    
    def get_feature_importance(self):
        """
        Aggregates feature importance across all base models.
        """
        pass
```

#### Expected Outputs:
- `src/bagging_ensemble.py` with working bagging implementation
- Comparable (or better) performance to single Decision Tree

---

### Phase 6: Baseline Models & Evaluation
**Duration:** Day 16-18  
**Goal:** Train all models, evaluate, and compare

#### Tasks:
1. Train baseline models using Spark MLlib:
   - LinearRegression
   - DecisionTreeRegressor
   - RandomForestRegressor (for comparison)
2. Evaluate all models using manual 10-fold CV
3. Train final models on full training set (80%)
4. Evaluate on holdout test set (20%)
5. Generate visualizations:
   - Model comparison bar chart
   - Feature importance plot
   - Predicted vs Actual scatter plot
   - Residual distribution

#### Key Metrics Table:
| Model | CV RMSE | CV MAE | CV R² | Test RMSE | Test MAE | Test R² |
|-------|---------|--------|-------|-----------|----------|---------|
| Linear Regression | | | | | | |
| Decision Tree | | | | | | |
| Random Forest | | | | | | |
| **Bagging (ours)** | | | | | | |

#### Expected Outputs:
- `src/evaluation.py` with metric functions
- All figures saved to `outputs/figures/`
- Complete results in notebook

---

### Phase 7: Documentation & Deliverables
**Duration:** Day 19-21  
**Goal:** Complete all deliverables

#### Tasks:
1. Finalize `README.md` with setup instructions
2. Write report (8-10 pages):
   - Section 1: Introduction and Problem Statement
   - Section 2: Dataset Description and Justification
   - Section 3: Methodology
   - Section 4: Results
   - Section 5: Discussion
   - Section 6: Conclusion
   - Section 7: References
3. Create presentation slides (10-15 slides)
4. Record 10-12 minute video
5. Upload video to YouTube (unlisted)
6. Final code review and cleanup
7. Zip everything for submission

#### Expected Outputs:
- `bigboyz_report.pdf`
- `bigboyz_slides.pptx`
- YouTube video link
- `bigboyz.zip` (final submission)

---

## Detailed Checklists

### ✅ Phase 1: Setup & Data Exploration
- [ ] Create directory structure
- [ ] Write `Dockerfile`
- [ ] Write `docker-compose.yml`
- [ ] Create `requirements.txt`
- [ ] Download `transactions.csv` from Dubai Pulse
- [ ] Create initial `bigboyz.ipynb`
- [ ] Load data and check shape
- [ ] Display schema and column types
- [ ] Check for missing values (count per column)
- [ ] Generate basic statistics (describe)
- [ ] Document initial observations

### ✅ Phase 2: Data Cleaning
- [ ] Create `src/data_ingestion.py`
- [ ] Create `src/data_cleaning.py`
- [ ] Define explicit schema for CSV loading
- [ ] Handle missing values in critical columns
- [ ] Remove duplicate rows
- [ ] Filter transaction types (keep sales only)
- [ ] Identify and handle price outliers
- [ ] Fix data type inconsistencies
- [ ] Verify cleaned data quality
- [ ] Save/checkpoint cleaned DataFrame

### ✅ Phase 3: Feature Engineering
- [ ] Create `src/feature_engineering.py`
- [ ] List all categorical columns
- [ ] Apply StringIndexer to categorical columns
- [ ] Apply OneHotEncoder (if needed)
- [ ] Extract year from transaction date
- [ ] Extract month from transaction date
- [ ] Extract quarter from transaction date
- [ ] Scale numerical features (StandardScaler)
- [ ] Create VectorAssembler for features
- [ ] Build full pipeline
- [ ] Apply pipeline to data
- [ ] Perform 80/20 train/test split
- [ ] Verify split sizes

### ✅ Phase 4: Manual Cross-Validation
- [ ] Create `src/cross_validation.py`
- [ ] Implement `create_folds()` function
- [ ] Implement `get_fold_split()` function
- [ ] Implement `calculate_metrics()` function (RMSE, MAE, R²)
- [ ] Implement `manual_cross_validate()` function
- [ ] Test with DecisionTreeRegressor
- [ ] Verify results are reasonable
- [ ] Add progress logging
- [ ] Document the implementation

### ✅ Phase 5: Bagging Ensemble
- [ ] Create `src/bagging_ensemble.py`
- [ ] Implement `bootstrap_sample()` function
- [ ] Create `BaggingRegressor` class
- [ ] Implement `__init__()` method
- [ ] Implement `fit()` method
- [ ] Implement `predict()` method
- [ ] Implement `get_feature_importance()` (optional)
- [ ] Test bagging on small data sample
- [ ] Run full training on training set
- [ ] Evaluate with manual CV
- [ ] Compare with single Decision Tree

### ✅ Phase 6: Evaluation
- [ ] Create `src/evaluation.py`
- [ ] Train LinearRegression baseline
- [ ] Train DecisionTreeRegressor baseline
- [ ] Train RandomForestRegressor baseline
- [ ] Run manual CV for all models
- [ ] Record CV metrics in table
- [ ] Train final models on full training set
- [ ] Evaluate on holdout test set
- [ ] Create model comparison bar chart
- [ ] Create feature importance plot
- [ ] Create predicted vs actual scatter plot
- [ ] Create residual distribution plot
- [ ] Save all figures to `outputs/figures/`
- [ ] Add all visualizations to notebook

### ✅ Phase 7: Documentation
- [ ] Complete README.md with:
  - [ ] Project description
  - [ ] Prerequisites
  - [ ] Installation steps
  - [ ] How to run
  - [ ] Project structure
  - [ ] Authors
- [ ] Write report:
  - [ ] Section 1: Introduction
  - [ ] Section 2: Dataset Description
  - [ ] Section 3: Methodology
  - [ ] Section 4: Results
  - [ ] Section 5: Discussion
  - [ ] Section 6: Conclusion
  - [ ] Section 7: References
- [ ] Create slides (10-15 slides)
- [ ] Record video (10-12 min)
- [ ] Upload to YouTube
- [ ] Final code cleanup
- [ ] Remove unnecessary files
- [ ] Create final zip file

---

## Timeline

### Week 1 (Days 1-7)
| Day | Focus | Tasks |
|-----|-------|-------|
| 1 | Setup | Directory structure, Dockerfile |
| 2 | Setup | Docker compose, download data |
| 3 | Explore | Data exploration, initial notebook |
| 4 | Clean | Load data with schema, missing values |
| 5 | Clean | Outliers, duplicates, filtering |
| 6 | Clean | Finalize cleaning pipeline |
| 7 | Features | Start feature engineering |

### Week 2 (Days 8-14)
| Day | Focus | Tasks |
|-----|-------|-------|
| 8 | Features | Categorical encoding |
| 9 | Features | Temporal features, scaling, vector assembly |
| 10 | CV | Start manual cross-validation |
| 11 | CV | Complete and test CV |
| 12 | CV | Debug and verify CV |
| 13 | Bagging | Start bagging implementation |
| 14 | Bagging | Complete and test bagging |

### Week 3 (Days 15-21)
| Day | Focus | Tasks |
|-----|-------|-------|
| 15 | Models | Train baseline models |
| 16 | Eval | Run all CV evaluations |
| 17 | Eval | Final evaluation on test set, visualizations |
| 18 | Docs | README, start report |
| 19 | Docs | Complete report |
| 20 | Present | Create slides, record video |
| 21 | Submit | Final review, zip, submit |

---

## Technical Specifications

### Docker Environment
```dockerfile
FROM jupyter/pyspark-notebook:latest

USER root

# Install additional system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /home/jovyan/work
```

### requirements.txt
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0  # Only for metrics comparison, not for main ML
```

### Spark Configuration
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("bigboyz-dubai-real-estate") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
```

---

## Evaluation Strategy

### Cross-Validation Approach
- 10-fold stratified by price range (optional) or random
- Each fold ~120K records (from 1.2M training)
- Report mean ± standard deviation for all metrics

### Metrics Definitions
```python
from pyspark.ml.evaluation import RegressionEvaluator

# RMSE - Root Mean Square Error
rmse_evaluator = RegressionEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="rmse"
)

# MAE - Mean Absolute Error
mae_evaluator = RegressionEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="mae"
)

# R² - Coefficient of Determination
r2_evaluator = RegressionEvaluator(
    labelCol="label", 
    predictionCol="prediction", 
    metricName="r2"
)
```

### Final Test Evaluation
- Train on full 80% training set
- Evaluate on 20% holdout (never seen during development)
- This is the "real" performance metric

---

## Deliverables Summary

| Deliverable | Format | Location |
|-------------|--------|----------|
| Main Notebook | `.ipynb` | `notebooks/bigboyz.ipynb` |
| Dockerfile | `Dockerfile` | `docker/Dockerfile` |
| Docker Compose | `.yml` | `docker/docker-compose.yml` |
| README | `.md` | `README.md` |
| Source Code | `.py` | `src/*.py` |
| Report | `.pdf` | `report/bigboyz_report.pdf` |
| Slides | `.pptx` | `presentation/bigboyz_slides.pptx` |
| Video | YouTube | `presentation/video_link.txt` |
| Final Submission | `.zip` | `bigboyz.zip` |

---

## Grading Alignment

| Criteria (from spec) | Weight | Our Approach |
|----------------------|--------|--------------|
| Problem Justification & Data Understanding | 20% | Clear business question, 1.5M records justifies Spark |
| Implementation | 40% | Docker ✓, Manual CV ✓, Bagging from scratch ✓ |
| Report | 25% | 8-10 pages, all required sections, screenshots |
| Presentation | 15% | Professional slides, 10-12 min video |

---

## Quick Commands Reference

### Start Docker Environment
```bash
cd bigboyz-dubai-real-estate
docker-compose up --build
```

### Run Notebook
```bash
# Access at http://localhost:8888
```

### Run Tests (if implemented)
```bash
docker exec -it bigboyz-spark pytest tests/
```

### Create Submission Zip
```bash
zip -r bigboyz.zip . -x "*.git*" -x "data/transactions.csv" -x "__pycache__/*"
```

---


Reference Data:
"https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open"
*Last Updated: January 22, 2026*

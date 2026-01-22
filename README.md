# bigboyz - Dubai Real Estate Price Prediction

**CSCI316: Big Data Mining Techniques and Implementation**
University of Wollongong in Dubai

---

## Project Overview

A large-scale machine learning pipeline to predict real estate transaction prices in Dubai using the Dubai Land Department's transactions dataset (~1.5 million records).

**Business Question:** *Can we accurately predict real estate transaction prices in Dubai based on property characteristics, location, and temporal factors?*

---

## Key Features

- **Manual 10-Fold Cross-Validation**: Implemented from scratch without using `CrossValidator` or `TrainValidationSplit`
- **Custom Bagging Ensemble**: Implemented from scratch without using `RandomForestRegressor`
- **Apache Spark (PySpark)**: All data processing at scale
- **Docker**: Containerized environment for reproducibility

---

## Project Structure

```
316-assignment1/
├── docker/
│   ├── Dockerfile              # Container definition
│   └── docker-compose.yml      # Multi-container orchestration
├── data/
│   └── transactions.csv        # Raw data (download separately)
├── notebooks/
│   └── bigboyz.ipynb           # Main deliverable notebook
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py       # Spark data loading utilities
│   ├── data_cleaning.py        # Preprocessing & cleaning
│   ├── feature_engineering.py  # Feature transformation pipeline
│   ├── cross_validation.py     # Manual 10-fold CV (FROM SCRATCH)
│   ├── bagging_ensemble.py     # Bagging implementation (FROM SCRATCH)
│   └── evaluation.py           # Metrics & visualization
├── outputs/
│   └── figures/                # Generated charts/plots
├── report/                     # Final report PDF
├── presentation/               # Slides and video link
├── CLAUDE.md                   # AI assistant context
├── PROJECT_PLAN.md             # Detailed implementation guide
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

---

## Prerequisites

- Docker & Docker Compose
- ~8GB RAM available for Spark
- ~2GB disk space for data

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repo-url>
cd 316-assignment1
```

### 2. Download the Dataset

The dataset is too large for GitHub. Download it manually:

1. Visit [Dubai Pulse - DLD Transactions](https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open)
2. Download the transactions CSV file
3. Place it in the `data/` folder as `transactions.csv`

### 3. Start the Docker Environment

```bash
cd docker
docker-compose up --build
```

### 4. Access Jupyter Notebook

Open your browser and navigate to:
```
http://localhost:8888
```

### 5. Run the Notebook

Open `notebooks/bigboyz.ipynb` and run all cells.

---

## Running Without Docker

If you prefer to run locally:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pyspark jupyter

# Start Jupyter
jupyter notebook notebooks/bigboyz.ipynb
```

---

## Source Code Modules

| Module | Description |
|--------|-------------|
| `data_ingestion.py` | Spark session creation, CSV loading with schema |
| `data_cleaning.py` | Missing value handling, outlier removal, filtering |
| `feature_engineering.py` | Categorical encoding, temporal features, vector assembly |
| `cross_validation.py` | **Manual 10-fold CV implementation (FROM SCRATCH)** |
| `bagging_ensemble.py` | **Custom Bagging Regressor (FROM SCRATCH)** |
| `evaluation.py` | Metrics calculation, comparison tables, visualizations |

---

## Models Implemented

### Baselines (using Spark MLlib)
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Custom Implementation (FROM SCRATCH)
- **Bagging Ensemble Regressor**
  - Base learner: `DecisionTreeRegressor` from MLlib
  - Bootstrap sampling: implemented manually
  - Prediction aggregation: averaging implemented manually

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Square Error |
| MAE | Mean Absolute Error |
| R² | Coefficient of Determination |

---

## Important Notes

### From-Scratch Requirements

This project implements the following **from scratch** as per assignment requirements:

1. **10-Fold Cross-Validation** (`src/cross_validation.py`)
   - Manual fold assignment
   - Manual train/validation splitting
   - Manual metric aggregation

2. **Bagging Ensemble** (`src/bagging_ensemble.py`)
   - Manual bootstrap sampling
   - Manual training of multiple trees
   - Manual prediction averaging

### What We CAN Use from MLlib

- `DecisionTreeRegressor` as base learners
- `RegressionEvaluator` for metrics
- `StringIndexer`, `VectorAssembler`, `StandardScaler` for feature engineering

### What We CANNOT Use

- `CrossValidator`
- `TrainValidationSplit`
- `ParamGridBuilder`
- `RandomForestRegressor` as the main ensemble model

---

## Troubleshooting

### Memory Issues
If you encounter memory errors, adjust Spark configuration in `src/data_ingestion.py`:
```python
.config("spark.driver.memory", "8g")
.config("spark.executor.memory", "8g")
```

### Data Loading Issues
Ensure your CSV file matches the expected schema. You may need to update column names in:
- `src/data_ingestion.py` - schema definition
- `notebooks/bigboyz.ipynb` - column name variables

---

## Authors

**Team bigboyz**
CSCI316 - Big Data Mining
University of Wollongong in Dubai

---

## License

This project is for educational purposes as part of CSCI316 coursework.

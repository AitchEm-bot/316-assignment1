# CLAUDE.md - Project Context for Claude Code

## Project Overview

**Project Name:** bigboyz - Dubai Real Estate Price Prediction  
**Course:** CSCI316: Big Data Mining Techniques and Implementation  
**University:** University of Wollongong in Dubai  
**Weight:** 15% of final grade  
**Deadline:** Friday of Week 6 (3 weeks from January 22, 2026)

---

## Problem Statement

Build a large-scale machine learning pipeline to predict real estate transaction prices in Dubai using the Dubai Land Department's transactions dataset (~1.5 million records).

**Business Question:** *"Can we accurately predict real estate transaction prices in Dubai based on property characteristics, location, and temporal factors?"*

---

## Dataset

- **Source:** Dubai Pulse - Dubai Land Department Transactions
- **URL:** https://www.dubaipulse.gov.ae/data/dld-transactions/dld_transactions-open
- **Size:** ~1.5 million transactions (2004-2025)
- **Format:** CSV
- **Target Variable:** Transaction price (AED)

---

## Critical Requirements (FROM PROJECT SPECIFICATION)

### Must Implement FROM SCRATCH (No Library Functions)

1. **10-Fold Cross-Validation**
   - Cannot use `CrossValidator` or `TrainValidationSplit` from Spark MLlib
   - Must manually split data into 10 folds
   - Must manually iterate through folds, train, evaluate, and aggregate metrics

2. **Bagging Ensemble Model**
   - Cannot use `RandomForestRegressor` as the ensemble (can use as baseline comparison)
   - Must implement bootstrap sampling logic yourself
   - Must implement prediction aggregation (averaging) yourself
   - CAN use `DecisionTreeRegressor` from MLlib as base learners

### Must Use

- **Apache Spark (PySpark)** for all data processing
- **Spark MLlib** for base model implementations
- **Docker** for containerization and reproducibility

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Data Processing | PySpark |
| ML Framework | Spark MLlib |
| Base Models | DecisionTreeRegressor (MLlib) |
| Ensemble | Custom Bagging (from scratch) |
| Cross-Validation | Custom 10-fold (from scratch) |
| Containerization | Docker |
| Notebook | Jupyter |

---

## Project Structure

```
bigboyz-dubai-real-estate/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   └── .gitkeep
├── notebooks/
│   └── bigboyz.ipynb            # MAIN DELIVERABLE
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── cross_validation.py      # FROM SCRATCH
│   ├── bagging_ensemble.py      # FROM SCRATCH
│   └── evaluation.py
├── outputs/
│   └── figures/
├── README.md
├── requirements.txt
├── CLAUDE.md
└── PROJECT_PLAN.md
```

---

## Data Split Strategy

```
Full Dataset (1.5M records)
         │
         ▼
┌────────────────────────┬────────────────────────┐
│   Training Set (80%)   │   Holdout Test (20%)   │
│      ~1.2M records     │     ~300K records      │
│                        │                        │
│   Used for:            │   Used for:            │
│   • 10-fold CV         │   • FINAL evaluation   │
│   • Model tuning       │   • Never touched      │
│   • Ensemble training  │     during training    │
└────────────────────────┴────────────────────────┘
```

---

## Models to Implement

### Baselines (using Spark MLlib)
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

### Custom Implementation (FROM SCRATCH)
4. Bagging Ensemble Regressor
   - Base learner: DecisionTreeRegressor (MLlib)
   - Number of estimators: configurable (default 10)
   - Bootstrap sampling: with replacement
   - Prediction: average of all base model predictions

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Square Error - penalizes large errors |
| MAE | Mean Absolute Error - average prediction error |
| R² | Coefficient of determination - variance explained |

---

## Deliverables Checklist

### Code Base (Zipped)
- [ ] `bigboyz.ipynb` - Main Jupyter notebook
- [ ] `Dockerfile` and `docker-compose.yml`
- [ ] `README.md` with setup instructions
- [ ] All source files in `src/`

### Report (PDF, 8-10 pages)
1. Introduction and Problem Statement
2. Dataset Description and Justification for Big Data Tools
3. Methodology (Preprocessing, Models, Implementation Details)
4. Results (Accuracy, Performance Metrics, Screenshots)
5. Discussion and Interpretation
6. Conclusion and Future Work
7. References

### Presentation
- [ ] PowerPoint/PDF slide deck
- [ ] 10-12 minute recorded video (YouTube upload)

---

## Code Style Guidelines

- Write clean, documented code with docstrings
- Include comments explaining the "from scratch" implementations
- Use meaningful variable names
- Keep functions modular and testable
- Add screenshots of outputs to the notebook for the report

---

## Important Notes for Claude Code

1. **Always verify Spark session is active** before DataFrame operations
2. **Do not use** `CrossValidator`, `TrainValidationSplit`, or `ParamGridBuilder` for cross-validation
3. **Do not use** `RandomForestRegressor` as the main ensemble model (only as baseline)
4. **Test incrementally** - verify each component works before moving to the next
5. **Generate visualizations** for the report as you go
6. **Document everything** - the professor will review the code

---

## Quick Reference: From-Scratch Implementations

### Manual 10-Fold CV Pseudocode
```python
def manual_k_fold_cv(data, model_class, k=10):
    folds = split_into_k_folds(data, k)
    metrics = []
    for i in range(k):
        val_fold = folds[i]
        train_folds = combine(folds except i)
        model = model_class.fit(train_folds)
        predictions = model.predict(val_fold)
        metrics.append(evaluate(predictions))
    return average(metrics)
```

### Bagging Pseudocode
```python
class BaggingRegressor:
    def fit(self, data, n_estimators=10):
        self.models = []
        for _ in range(n_estimators):
            sample = bootstrap_sample(data)  # WITH replacement
            tree = DecisionTreeRegressor().fit(sample)
            self.models.append(tree)
    
    def predict(self, data):
        all_preds = [m.predict(data) for m in self.models]
        return average(all_preds)
```

---

## Contact & Resources

- **Dubai Pulse Data:** https://www.dubaipulse.gov.ae
- **PySpark Docs:** https://spark.apache.org/docs/latest/api/python/
- **Spark MLlib Guide:** https://spark.apache.org/docs/latest/ml-guide.html

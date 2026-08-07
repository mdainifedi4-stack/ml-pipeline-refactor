# System Architecture

## Old System (Flask App)

A monolithic web application where:
- Routes handled HTTP requests
- Global variables stored state
- Plotting, metrics, and training code was copy-pasted across 6+ route handlers
- No experiment tracking, no versioning, no automation

## New System (This Pipeline)

### Design Principles

1. **No global state** — all data flows through function arguments
2. **Single responsibility** — each module does one thing
3. **Testability** — every function has a unit test
4. **Reproducibility** — MLflow logs every parameter, metric, and artifact

### Data Flow

1. **Ingestion** — load_data supports CSV and Excel
2. **Preprocessing** — clean_data imputes missing values, label-encodes categoricals
3. **Problem Detection** — auto-detects classification vs regression
4. **Split & Scale** — train_test_split + StandardScaler
5. **Model Training** — 6 models per problem type with GridSearchCV
6. **Evaluation** — classification or regression metrics
7. **Plotting** — correlation, confusion matrix, feature importance, residuals
8. **Champion Selection** — best F1 for classification, best R² for regression
9. **Tracking** — MLflow logs params, metrics, artifacts, models

### Module Dependencies
pipeline.py
├── preprocessing.py
├── training.py
│   ├── metrics.py
│   └── (sklearn)
├── plots.py
│   └── (matplotlib, seaborn)
├── utils/io_utils.py
└── tracking/mlflow_tracker.py
└── (mlflow)
# AutoML Pipeline Pro: From Raw Data to Deployed Model

An end-to-end automated ML pipeline with MLOps integration. Built by refactoring a monolithic Flask web app into a modular, containerized, experiment-tracked system.

## What This Project Does

| Before (Flask App) | After (This Pipeline) |
|---|---|
| Manual CSV upload via web UI | Automated batch processing |
| Code copy-pasted across 6+ routes | Reusable, tested modules |
| Global mutable state | Pure functions, no side effects |
| No tracking or versioning | MLflow experiment tracking |
| Runs only on developer machine | Docker container, runs anywhere |

## Architecture

```
Raw Data → Ingestion → Preprocessing → Split/Scale → Model Loop (6 models, GridSearchCV)
                                                              ↓
                                                   Champion Selection (best F1/R²)
                                                              ↓
                                          MLflow Tracking ← Save Model + Plots
```

## Tech Stack

- Python 3.11, scikit-learn, pandas, numpy
- matplotlib, seaborn
- MLflow (SQLite backend)
- Docker & Docker Compose
- pytest

## Quick Start

### Local (Windows)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "."
python tests/test_pipeline.py
```

### Docker

```powershell
docker-compose up --build
```

## Project Structure

```
ml-pipeline-refactor/
├── src/plots/           # Reusable plotting functions
├── src/metrics/         # Classification, regression, clustering metrics
├── src/preprocessing/   # Data loading, cleaning, problem-type detection
├── src/training/        # Split/scale, train, grid search, evaluate
├── src/utils/           # Timestamps, directories, model I/O
├── src/tracking/        # MLflow wrapper
├── src/pipeline/        # End-to-end orchestration
├── tests/               # pytest unit tests for every module
├── docs/                # Architecture docs and future work
├── legacy/              # Original Flask app (for reference)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Skills Demonstrated

- **Software Engineering:** Modular code, error handling, pure functions
- **ML Engineering:** Automated pipeline, model serialization, champion selection
- **MLOps:** Experiment tracking (MLflow), containerization (Docker)
- **Data Engineering:** Automated ETL, data validation
- **Data Science:** Feature engineering, model selection, hyperparameter tuning

## Future Work

See `docs/FUTURE_WORK.md` for detailed plans on DVC, Prefect/Airflow, Great Expectations, CI/CD, and clustering.

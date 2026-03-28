## Features
- Caches and creates CSV files for the involved data.

## How It Works

## 1. Data Ingestion
- Historical F1 data (2018–2024) is cached and extracted using FastF1.
- Generates multiple CSV files, including:
  - Race results
  - Qualifying results
  - Weather
  - Race Events
  - Starting tyre compounds
- Ensures the model has clean, structured input for analysis.

## Future Plans
- Process the ingested data to engineer features for prediction.
- Train a machine learning model (XGBoost - Monte Carlo hybrid model) to forecast race outcomes.
- Visualize model predictions with interactive charts and dashboards.

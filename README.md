# F1 Race Predictor

A hybrid XGBoost + Monte Carlo model that predicts the full finishing order of Formula 1 races using pre-race data only. Built on FastF1 historical data from 2018–2024.

[Access Formula Forecast Here](https://formulaforecast.vercel.app)

---

## How It Works

The model runs in two stages:

**Stage 1 — XGBoost Ranking Model**
Trained on 7 seasons of historical race data, the model learns which pre-race signals (qualifying gap, recent form, circuit history, championship pressure) are most predictive of finishing order. It uses XGBoost's `rank:pairwise` objective, which learns who beats who rather than predicting raw position numbers.

**Stage 2 — Monte Carlo Simulation**
XGBoost's driver strength scores become the baseline for 10,000 race simulations. Each simulation independently draws from probability distributions for DNF events, safety car deployments, and rain — capturing the inherent randomness of a race. The output is a probability distribution over all possible finishing positions for every driver.

**Key constraint:** The model uses only information available before the race starts — qualifying position, gap to pole, starting tyre compound, weather at race start, and historical statistics. No in-race data (live tyre degradation, safety car timing, pit strategy) is used.

---

## Project Structure

```
f1-predictor/
│
├── data_ingestion.py       # Pull raw data from FastF1 (run once)
├── data_processing.py      # Build features from raw CSVs
├── model.py                # Train and evaluate the XGBoost model
├── simulate.py             # Monte Carlo simulation engine
├── predict.py              # Predict an upcoming race
├── index.html            # Browser dashboard to visualise predictions
│
├── data/                   # CSVs produced by ingestion and processing
│   ├── race_results.csv
│   ├── qualifying_results.csv
│   ├── weather_snapshots.csv
│   ├── race_control.csv
│   ├── starting_stints.csv
│   └── processed_data.csv
│
├── models/                 # Saved model files
│   ├── xgb_f1_ranker.ubj
│   └── feature_cols.pkl
│
└── ff1_cache/              # FastF1 local cache (auto-created)
```

---

## Requirements

```bash
pip install fastf1 pandas numpy xgboost scikit-learn scipy joblib tqdm
```

Python 3.9 or higher recommended.

---

## Setup & Usage

### Step 1 — Pull the data

```bash
python data_ingestion.py
```

Downloads 2018–2024 race results, qualifying times, weather snapshots, race control events, and starting tyre compounds from FastF1. Saves 5 CSVs to `./data/`.

**First run takes 30–90 minutes.** FastF1 caches all responses locally so subsequent runs are near-instant. If a session fails it logs a warning and continues — a single bad session won't crash the run.

Already have some CSVs? The script skips any file that already exists on disk.

### Step 2 — Process the data

```bash
python data_processing.py
```

Joins the 5 raw CSVs and engineers all model features. Outputs a single `processed_data.csv` with ~47 columns per driver per race. Takes under a minute.

Features built in this step:

| Feature Group | Examples |
|---|---|
| Rolling driver form | Avg finish, avg grid, DNF rate over last 5 races |
| Rolling team form | Avg finish, DNF rate over last 5 races |
| Circuit history | Driver win rate, podium rate, avg finish at this circuit |
| Championship context | Points gap to leader, championship position pre-race |
| Safety car probability | Historical SC and VSC rate per circuit |
| Encoded categoricals | Compound, circuit, team, driver as integers |

### Step 3 — Train the model

```bash
python model.py
```

Trains the XGBoost ranking model and evaluates it. Data is split by regulation era — not randomly — to avoid data leakage:

- **Train:** 2018–2021
- **Validation:** 2022–2023
- **Test:** 2024

Prints Spearman rank correlation, winner accuracy, and podium overlap for both validation and test sets. Also runs 5-fold cross-validation and prints the top 15 most predictive features. Saves the trained model to `./models/`.

### Step 4 — Predict a race

```bash
python predict.py 2025 4
```

Predicts the full finishing order for a given year and round number. Run this **after qualifying on Saturday** — the model needs qualifying times as input.

The script fetches fresh qualifying results, weather, and starting compounds from FastF1, rebuilds all rolling and historical features on the fly, runs 10,000 simulations, and saves two output files:

```
data/predict_2025_R4_summary.csv         ← predicted order + probabilities
data/predict_2025_R4_probabilities.csv   ← full 20×20 position matrix
```

You can also call it from Python:

```python
from predict import predict
summary, prob_table = predict(year=2025, round_num=4)
```

### Step 5 — View results

Open `results.html` in any browser. Drag and drop the two output CSVs from Step 4 into the file slots and click **Load Results**.

The dashboard shows:
- Predicted winner, win probability, pole sitter, highest DNF risk
- Podium cards with animated win / podium / points probability bars
- Full 20-driver table with mean predicted position and uncertainty
- 20×20 probability matrix heatmap

To preview without running the model first, click **"or load demo data"** inside the dashboard.

---

## Simulating a Past Race

To run the simulation on a race already in your processed data (for backtesting or reviewing past predictions):

```python
from simulate import simulate_race
summary, prob_table = simulate_race(year=2024, round_num=5)
```

---

## Accuracy

Evaluated on the 2024 season (held-out test set):

| Metric | Score |
|---|---|
| Spearman rank correlation | ~0.78–0.82 |
| Winner prediction accuracy | ~50–58% |
| Avg podium overlap | ~65–72% |

Accuracy is lower in seasons with major regulation changes (2022 was a new era). The model is retrained within era to account for this — do not mix pre-2022 and post-2022 training data.

---

## Data Sources

All data comes from [FastF1](https://github.com/theOehrly/Fast-F1), which wraps the official F1 timing API and the Ergast/Jolpica historical database. No API key is required. Historical data is available from 2018 onwards for telemetry and timing; results and schedule data go back to 1950 via Ergast.

---

## Limitations

- **DNF causes are not classified.** FastF1 marks a driver as DNF but doesn't reliably distinguish engine failure from accident. DNF probabilities are based on overall historical rates.
- **Pre-2018 granular data is unavailable.** Lap times, tyre data, and weather before 2018 don't exist in any public API.
- **Regulation change seasons are harder to predict.** 2022 and 2026 (new technical regulations) significantly shift team performance hierarchies. Expect lower accuracy in the first half of those seasons.
- **Qualifying is required.** `predict.py` cannot run before qualifying completes on Saturday. Grid position and gap to pole are the two strongest model features.
- **Starting compounds are announced pre-race but confirmed in the data post-qualifying.** There is occasionally a short lag before FastF1 reflects the official Pirelli compound nominations.

---

## File Run Order

```
data_ingestion.py  →  data_processing.py  →  model.py  →  predict.py  →  results.html
```

`simulate.py` is used internally by `predict.py` and does not need to be run directly unless backtesting a past race.

import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import fastf1

from simulate import run_simulation, print_summary, N_SIMULATIONS

warnings.filterwarnings("ignore")

MODEL_DIR  = "./models"
INPUT_DIR  = "./data"
OUTPUT_DIR = "./data"

fastf1.Cache.enable_cache("./ff1_cache")


def load_model():
    model = xgb.Booster()
    model.load_model(os.path.join(MODEL_DIR, "xgb_f1_ranker.ubj"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, feature_cols


def load_historical(processed_path=None):
    if processed_path is None:
        processed_path = os.path.join(INPUT_DIR, "processed_data.csv")
    return pd.read_csv(processed_path)


def get_qualifying_data(year, round_num):
    session = fastf1.get_session(year, round_num, "Q")
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    results = session.results

    pole_row = results[results["Position"] == 1]
    if pole_row.empty:
        raise ValueError("Could not find pole position in qualifying results")

    def best_q_time(row):
        for col in ["Q3", "Q2", "Q1"]:
            try:
                t = row.get(col)
                if t and pd.notna(t):
                    return t.total_seconds()
            except Exception:
                pass
        return None

    pole_time = best_q_time(pole_row.iloc[0])

    rows = []
    for _, driver in results.iterrows():
        best_time = best_q_time(driver)
        gap = round(best_time - pole_time, 4) if best_time and pole_time else None

        rows.append({
            "driver_number":  str(driver["DriverNumber"]),
            "abbreviation":   driver["Abbreviation"],
            "full_name":      driver["FullName"],
            "team_name":      driver["TeamName"],
            "grid_position":  int(driver["Position"]),
            "gap_to_pole_s":  gap,
            "best_q_time_s":  best_time,
            "q1_time_s":      best_q_time({"Q1": driver.get("Q1"), "Q2": None, "Q3": None}),
            "q2_time_s":      best_q_time({"Q1": driver.get("Q2"), "Q2": None, "Q3": None}),
            "q3_time_s":      best_q_time({"Q1": driver.get("Q3"), "Q2": None, "Q3": None}),
        })

    return pd.DataFrame(rows)


def get_weather_data(year, round_num):
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=False, telemetry=False, weather=True, messages=False)

    w = session.weather_data
    if w is None or w.empty:
        return {}

    snap = w.iloc[0]
    return {
        "air_temp_c":    snap.get("AirTemp"),
        "track_temp_c":  snap.get("TrackTemp"),
        "humidity_pct":  snap.get("Humidity"),
        "pressure_mbar": snap.get("Pressure"),
        "wind_speed_ms": snap.get("WindSpeed"),
        "wind_dir_deg":  snap.get("WindDirection"),
        "rainfall":      int(bool(snap.get("Rainfall", 0))),
    }


def get_starting_compounds(year, round_num):
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = session.laps
    stint1 = (
        laps[laps["LapNumber"] == 1][["DriverNumber", "Compound", "TyreLife", "FreshTyre"]]
        .drop_duplicates(subset="DriverNumber")
    )

    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}

    result = {}
    for _, row in stint1.iterrows():
        result[str(row["DriverNumber"])] = {
            "start_compound":         row.get("Compound"),
            "start_compound_encoded": compound_map.get(row.get("Compound"), -1),
            "tyre_life_start":        row.get("TyreLife"),
            "fresh_tyre":             row.get("FreshTyre"),
        }
    return result


def build_rolling_features(driver_number, abbreviation, team_name, hist_df, circuit, year, round_num):
    driver_hist = hist_df[
        (hist_df["abbreviation"] == abbreviation) &
        ((hist_df["year"] < year) | ((hist_df["year"] == year) & (hist_df["round"] < round_num)))
    ].sort_values(["year", "round"]).tail(5)

    team_hist = hist_df[
        (hist_df["team_name"] == team_name) &
        ((hist_df["year"] < year) | ((hist_df["year"] == year) & (hist_df["round"] < round_num)))
    ].sort_values(["year", "round"]).tail(5)

    circuit_hist = hist_df[
        (hist_df["abbreviation"] == abbreviation) &
        (hist_df["circuit"] == circuit) &
        ((hist_df["year"] < year) | ((hist_df["year"] == year) & (hist_df["round"] < round_num)))
    ]

    team_circuit_hist = hist_df[
        (hist_df["team_name"] == team_name) &
        (hist_df["circuit"] == circuit) &
        ((hist_df["year"] < year) | ((hist_df["year"] == year) & (hist_df["round"] < round_num)))
    ]

    rolling_avg_finish       = driver_hist["finish_position"].mean() if len(driver_hist) > 0 else np.nan
    rolling_avg_grid         = driver_hist["grid_position"].mean() if len(driver_hist) > 0 else np.nan
    rolling_dnf_rate         = driver_hist["dnf"].mean() if len(driver_hist) > 0 else 0.05
    rolling_avg_gap_to_pole  = driver_hist["gap_to_pole_s"].mean() if len(driver_hist) > 0 else np.nan

    team_rolling_avg_finish  = team_hist["finish_position"].mean() if len(team_hist) > 0 else np.nan
    team_rolling_dnf_rate    = team_hist["dnf"].mean() if len(team_hist) > 0 else 0.05

    driver_circuit_avg_finish  = circuit_hist["finish_position"].mean() if len(circuit_hist) > 0 else np.nan
    driver_circuit_win_rate    = (circuit_hist["finish_position"] == 1).mean() if len(circuit_hist) > 0 else np.nan
    driver_circuit_podium_rate = (circuit_hist["finish_position"] <= 3).mean() if len(circuit_hist) > 0 else np.nan
    driver_circuit_dnf_rate    = circuit_hist["dnf"].mean() if len(circuit_hist) > 0 else np.nan

    team_circuit_avg_finish    = team_circuit_hist["finish_position"].mean() if len(team_circuit_hist) > 0 else np.nan
    team_circuit_win_rate      = (team_circuit_hist["finish_position"] == 1).mean() if len(team_circuit_hist) > 0 else np.nan

    return {
        "rolling_avg_finish":          rolling_avg_finish,
        "rolling_avg_grid":            rolling_avg_grid,
        "rolling_dnf_rate":            rolling_dnf_rate,
        "rolling_avg_gap_to_pole":     rolling_avg_gap_to_pole,
        "team_rolling_avg_finish":     team_rolling_avg_finish,
        "team_rolling_dnf_rate":       team_rolling_dnf_rate,
        "driver_circuit_avg_finish":   driver_circuit_avg_finish,
        "driver_circuit_win_rate":     driver_circuit_win_rate,
        "driver_circuit_podium_rate":  driver_circuit_podium_rate,
        "driver_circuit_dnf_rate":     driver_circuit_dnf_rate,
        "team_circuit_avg_finish":     team_circuit_avg_finish,
        "team_circuit_win_rate":       team_circuit_win_rate,
    }


def build_championship_features(abbreviation, team_name, hist_df, year, round_num):
    season_hist = hist_df[
        (hist_df["year"] == year) &
        (hist_df["round"] < round_num)
    ]

    driver_points = (
        hist_df[
            (hist_df["abbreviation"] == abbreviation) &
            (hist_df["year"] == year) &
            (hist_df["round"] < round_num)
        ]["points"].sum()
    )

    if season_hist.empty:
        return {
            "cumulative_points":         driver_points,
            "championship_gap_to_leader": 0.0,
            "championship_position":     1.0,
        }

    standings = (
        season_hist.groupby("abbreviation")["points"]
        .sum()
        .reset_index()
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )

    leader_points = standings["points"].max()
    gap = leader_points - driver_points

    pos_row = standings[standings["abbreviation"] == abbreviation]
    position = float(pos_row.index[0] + 1) if not pos_row.empty else float(len(standings) + 1)

    return {
        "cumulative_points":          driver_points,
        "championship_gap_to_leader": gap,
        "championship_position":      position,
    }


def build_sc_probability(circuit, hist_df):
    circuit_hist = hist_df[hist_df["circuit"] == circuit]

    sc_prob  = circuit_hist["sc_probability"].mean()  if "sc_probability"  in circuit_hist.columns else 0.30
    vsc_prob = circuit_hist["vsc_probability"].mean() if "vsc_probability" in circuit_hist.columns else 0.20

    return {
        "sc_probability":  sc_prob  if pd.notna(sc_prob)  else 0.30,
        "vsc_probability": vsc_prob if pd.notna(vsc_prob) else 0.20,
    }


def encode_categoricals(race_df, hist_df):
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2, "INTERMEDIATE": 3, "WET": 4}
    race_df["start_compound_encoded"] = race_df["start_compound"].map(compound_map).fillna(-1).astype(int)

    all_circuits = hist_df["circuit"].unique().tolist()
    circuit = race_df["circuit"].iloc[0]
    race_df["circuit_encoded"] = all_circuits.index(circuit) if circuit in all_circuits else -1

    all_teams = hist_df["team_name"].unique().tolist()
    race_df["team_encoded"] = race_df["team_name"].apply(
        lambda t: all_teams.index(t) if t in all_teams else -1
    )

    all_drivers = hist_df["abbreviation"].unique().tolist()
    race_df["driver_encoded"] = race_df["abbreviation"].apply(
        lambda d: all_drivers.index(d) if d in all_drivers else -1
    )

    return race_df


def build_race_df(year, round_num, event_name, circuit, country, hist_df):
    print("  Fetching qualifying results...")
    quali_df = get_qualifying_data(year, round_num)

    print("  Fetching weather snapshot...")
    weather  = get_weather_data(year, round_num)

    print("  Fetching starting tyre compounds...")
    compounds = get_starting_compounds(year, round_num)

    sc_probs = build_sc_probability(circuit, hist_df)

    regulation_era        = 0 if year <= 2021 else 1
    race_number_in_season = round_num

    rows = []
    for _, driver in quali_df.iterrows():
        drv_num = str(driver["driver_number"])
        abbr    = driver["abbreviation"]
        team    = driver["team_name"]

        rolling = build_rolling_features(drv_num, abbr, team, hist_df, circuit, year, round_num)
        champ   = build_championship_features(abbr, team, hist_df, year, round_num)
        tyre    = compounds.get(drv_num, {
            "start_compound": None, "start_compound_encoded": -1,
            "tyre_life_start": 0,   "fresh_tyre": True
        })

        row = {
            "year":                   year,
            "round":                  round_num,
            "event_name":             event_name,
            "circuit":                circuit,
            "country":                country,
            "driver_number":          drv_num,
            "abbreviation":           abbr,
            "full_name":              driver["full_name"],
            "team_name":              team,
            "grid_position":          driver["grid_position"],
            "gap_to_pole_s":          driver["gap_to_pole_s"],
            "best_q_time_s":          driver["best_q_time_s"],
            "start_compound":         tyre.get("start_compound"),
            "start_compound_encoded": tyre.get("start_compound_encoded", -1),
            "tyre_life_start":        tyre.get("tyre_life_start", 0),
            "fresh_tyre":             tyre.get("fresh_tyre", True),
            "regulation_era":         regulation_era,
            "race_number_in_season":  race_number_in_season,
            "dnf":                    0,
            "finish_position":        np.nan,
            **rolling,
            **champ,
            **sc_probs,
            **weather,
        }
        rows.append(row)

    race_df = pd.DataFrame(rows)
    race_df = encode_categoricals(race_df, hist_df)
    return race_df


def predict(year, round_num, event_name=None, circuit=None, country=None):
    print(f"\n{'=' * 60}")
    print(f"F1 Race Predictor  |  {year} Round {round_num}")
    print(f"{'=' * 60}")

    model, feature_cols = load_model()
    hist_df = load_historical()

    if event_name is None or circuit is None:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        event = schedule[schedule["RoundNumber"] == round_num].iloc[0]
        event_name = event_name or event["EventName"]
        circuit    = circuit    or event["Location"]
        country    = country    or event["Country"]

    print(f"\n  Event   : {event_name}")
    print(f"  Circuit : {circuit}, {country}")

    print("\nBuilding feature set...")
    race_df = build_race_df(year, round_num, event_name, circuit, country, hist_df)

    print(f"\nRunning {N_SIMULATIONS:,} simulations...")
    summary, prob_table, _ = run_simulation(race_df, model, feature_cols)

    print_summary(summary, event_name, N_SIMULATIONS)

    summary_path = os.path.join(OUTPUT_DIR, f"predict_{year}_R{round_num}_summary.csv")
    prob_path    = os.path.join(OUTPUT_DIR, f"predict_{year}_R{round_num}_probabilities.csv")

    summary.to_csv(summary_path, index=False)
    prob_table.to_csv(prob_path)

    print(f"\n  Summary saved     : {summary_path}")
    print(f"  Probabilities saved: {prob_path}")
    print(f"\n{'=' * 60}")
    print("✅  Prediction complete.")
    print(f"{'=' * 60}")

    return summary, prob_table


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        year      = int(sys.argv[1])
        round_num = int(sys.argv[2])
    else:
        print("Usage: python predict.py <year> <round>")
        print("Example: python predict.py 2025 4")
        print("\nNo arguments provided — running example with 2024 R1...\n")
        year      = 2024
        round_num = 1

    predict(year, round_num)
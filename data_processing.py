import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

INPUT_DIR = "./data"
OUTPUT_DIR = "./data"
ROLLING_WINDOW = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    race = pd.read_csv(os.path.join(INPUT_DIR, "race_results.csv"))
    quali = pd.read_csv(os.path.join(INPUT_DIR, "qualifying_results.csv"))
    weather = pd.read_csv(os.path.join(INPUT_DIR, "weather_snapshots.csv"))
    stints = pd.read_csv(os.path.join(INPUT_DIR, "starting_stints.csv"))
    rc = pd.read_csv(os.path.join(INPUT_DIR, "race_control.csv"))
    return race, quali, weather, stints, rc


def merge_base(race, quali, weather, stints):
    df = race.merge(
        quali[["year", "round", "driver_number", "gap_to_pole_s",
               "q1_time_s", "q2_time_s", "q3_time_s", "best_q_time_s"]],
        on=["year", "round", "driver_number"],
        how="left"
    )
    df = df.merge(
        weather[["year", "round", "air_temp_c", "track_temp_c",
                 "humidity_pct", "pressure_mbar", "wind_speed_ms",
                 "wind_dir_deg", "rainfall"]],
        on=["year", "round"],
        how="left"
    )
    df = df.merge(
        stints[["year", "round", "driver_number",
                "start_compound", "tyre_life_start", "fresh_tyre"]],
        on=["year", "round", "driver_number"],
        how="left"
    )
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    return df


def add_rolling_driver_features(df):
    df = df.sort_values(["driver_number", "year", "round"]).reset_index(drop=True)

    def rolling_mean_excl_current(series):
        return series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()

    df["rolling_avg_finish"] = (
        df.groupby("driver_number")["finish_position"]
        .transform(rolling_mean_excl_current)
    )
    df["rolling_avg_grid"] = (
        df.groupby("driver_number")["grid_position"]
        .transform(rolling_mean_excl_current)
    )
    df["rolling_dnf_rate"] = (
        df.groupby("driver_number")["dnf"]
        .transform(rolling_mean_excl_current)
    )
    df["rolling_avg_gap_to_pole"] = (
        df.groupby("driver_number")["gap_to_pole_s"]
        .transform(rolling_mean_excl_current)
    )

    return df


def add_rolling_team_features(df):
    df = df.sort_values(["team_name", "year", "round"]).reset_index(drop=True)

    def rolling_mean_excl_current(series):
        return series.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()

    df["team_rolling_avg_finish"] = (
        df.groupby("team_name")["finish_position"]
        .transform(rolling_mean_excl_current)
    )
    df["team_rolling_dnf_rate"] = (
        df.groupby("team_name")["dnf"]
        .transform(rolling_mean_excl_current)
    )

    return df


def add_circuit_history_features(df):
    rows = []

    for idx, row in df.iterrows():
        year = row["year"]
        round_ = row["round"]
        driver = row["driver_number"]
        team = row["team_name"]
        circuit = row["circuit"]

        past = df[
            (df["circuit"] == circuit) &
            ((df["year"] < year) | ((df["year"] == year) & (df["round"] < round_)))
        ]

        driver_past = past[past["driver_number"] == driver]
        team_past = past[past["team_name"] == team]

        driver_avg = driver_past["finish_position"].mean() if len(driver_past) > 0 else np.nan
        driver_win_rate = (driver_past["finish_position"] == 1).mean() if len(driver_past) > 0 else np.nan
        driver_podium_rate = (driver_past["finish_position"] <= 3).mean() if len(driver_past) > 0 else np.nan
        driver_circuit_dnf_rate = driver_past["dnf"].mean() if len(driver_past) > 0 else np.nan

        team_avg = team_past["finish_position"].mean() if len(team_past) > 0 else np.nan
        team_win_rate = (team_past["finish_position"] == 1).mean() if len(team_past) > 0 else np.nan

        rows.append({
            "idx": idx,
            "driver_circuit_avg_finish": driver_avg,
            "driver_circuit_win_rate": driver_win_rate,
            "driver_circuit_podium_rate": driver_podium_rate,
            "driver_circuit_dnf_rate": driver_circuit_dnf_rate,
            "team_circuit_avg_finish": team_avg,
            "team_circuit_win_rate": team_win_rate,
        })

    circuit_df = pd.DataFrame(rows).set_index("idx")
    df = df.join(circuit_df)
    return df


def add_championship_features(df):
    df = df.sort_values(["year", "round"]).reset_index(drop=True)

    df["cumulative_points"] = df.groupby(["year", "driver_number"])["points"].cumsum().shift(1).fillna(0)

    def points_gap_to_leader(group):
        return group["cumulative_points"].max() - group["cumulative_points"]

    df["championship_gap_to_leader"] = (
        df.groupby(["year", "round"], group_keys=False)
        .apply(points_gap_to_leader)
    )

    df["championship_position"] = (
        df.groupby(["year", "round"])["cumulative_points"]
        .rank(ascending=False, method="min")
    )

    return df


def add_sc_probability(df, rc):
    sc_messages = rc[
        rc["message"].str.contains("SAFETY CAR", case=False, na=False) &
        ~rc["message"].str.contains("VIRTUAL", case=False, na=False)
    ]
    vsc_messages = rc[
        rc["message"].str.contains("VIRTUAL SAFETY CAR", case=False, na=False)
    ]

    sc_races = sc_messages[["circuit", "year", "round"]].drop_duplicates()
    vsc_races = vsc_messages[["circuit", "year", "round"]].drop_duplicates()

    all_races = rc[["circuit", "year", "round"]].drop_duplicates()

    sc_prob = (
        sc_races.groupby("circuit").size() /
        all_races.groupby("circuit").size()
    ).rename("sc_probability")

    vsc_prob = (
        vsc_races.groupby("circuit").size() /
        all_races.groupby("circuit").size()
    ).rename("vsc_probability")

    df = df.merge(sc_prob.reset_index(), on="circuit", how="left")
    df = df.merge(vsc_prob.reset_index(), on="circuit", how="left")

    df["sc_probability"] = df["sc_probability"].fillna(0.3)
    df["vsc_probability"] = df["vsc_probability"].fillna(0.2)

    return df


def add_regulation_era(df):
    df["regulation_era"] = df["year"].apply(lambda y: 0 if y <= 2021 else 1)
    return df


def add_race_number_in_season(df):
    df["race_number_in_season"] = df.groupby("year")["round"].rank(method="dense").astype(int)
    return df


def encode_categoricals(df):
    compound_map = {"SOFT": 0, "MEDIUM": 1, "HARD": 2,
                    "INTERMEDIATE": 3, "WET": 4}
    df["start_compound_encoded"] = df["start_compound"].map(compound_map).fillna(-1).astype(int)

    circuit_codes, _ = pd.factorize(df["circuit"])
    df["circuit_encoded"] = circuit_codes

    team_codes, _ = pd.factorize(df["team_name"])
    df["team_encoded"] = team_codes

    driver_codes, _ = pd.factorize(df["abbreviation"])
    df["driver_encoded"] = driver_codes

    return df


def select_final_features(df):
    feature_cols = [
        "year", "round", "event_name", "circuit", "country",
        "driver_number", "abbreviation", "full_name", "team_name",

        "finish_position",

        "grid_position",
        "gap_to_pole_s",
        "best_q_time_s",
        "start_compound_encoded",
        "tyre_life_start",
        "fresh_tyre",

        "rolling_avg_finish",
        "rolling_avg_grid",
        "rolling_dnf_rate",
        "rolling_avg_gap_to_pole",

        "team_rolling_avg_finish",
        "team_rolling_dnf_rate",

        "driver_circuit_avg_finish",
        "driver_circuit_win_rate",
        "driver_circuit_podium_rate",
        "driver_circuit_dnf_rate",
        "team_circuit_avg_finish",
        "team_circuit_win_rate",

        "cumulative_points",
        "championship_gap_to_leader",
        "championship_position",

        "sc_probability",
        "vsc_probability",

        "regulation_era",
        "race_number_in_season",

        "air_temp_c",
        "track_temp_c",
        "humidity_pct",
        "pressure_mbar",
        "wind_speed_ms",
        "wind_dir_deg",
        "rainfall",

        "circuit_encoded",
        "team_encoded",
        "driver_encoded",

        "dnf",
        "status",
        "points",
    ]

    available = [c for c in feature_cols if c in df.columns]
    return df[available]


if __name__ == "__main__":
    print("=" * 60)
    print("F1 Data Processing  |  2018–2024")
    print("=" * 60)

    print("Loading CSVs...")
    race, quali, weather, stints, rc = load_data()

    print("Merging base tables...")
    df = merge_base(race, quali, weather, stints)

    print("Adding rolling driver features...")
    df = add_rolling_driver_features(df)

    print("Adding rolling team features...")
    df = add_rolling_team_features(df)

    print("Adding circuit history features...")
    df = add_circuit_history_features(df)

    print("Adding championship features...")
    df = add_championship_features(df)

    print("Adding safety car probabilities...")
    df = add_sc_probability(df, rc)

    print("Adding regulation era...")
    df = add_regulation_era(df)

    print("Adding race number in season...")
    df = add_race_number_in_season(df)

    print("Encoding categoricals...")
    df = encode_categoricals(df)

    print("Selecting final features...")
    df = select_final_features(df)

    path = os.path.join(OUTPUT_DIR, "processed_data.csv")
    df.to_csv(path, index=False)

    print(f"\n✅  processed_data.csv  →  {len(df)} rows  x  {len(df.columns)} columns")
    print(f"    Saved to: {os.path.abspath(path)}")
    print("=" * 60)
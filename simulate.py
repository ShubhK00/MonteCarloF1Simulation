import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

MODEL_DIR  = "./models"
INPUT_DIR  = "./data"
OUTPUT_DIR = "./data"

N_SIMULATIONS = 10000
RANDOM_SEED   = 42

rng = np.random.default_rng(RANDOM_SEED)


def load_model():
    model        = xgb.Booster()
    model.load_model(os.path.join(MODEL_DIR, "xgb_f1_ranker.ubj"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, feature_cols


def get_base_scores(model, race_df, feature_cols):
    available = [c for c in feature_cols if c in race_df.columns]
    X = race_df[available].fillna(race_df[available].median())
    dmat = xgb.DMatrix(X)
    return model.predict(dmat)


def simulate_dnf(race_df, n_sims):
    driver_dnf_rate = race_df["rolling_dnf_rate"].fillna(0.05).values
    team_dnf_rate   = race_df["team_rolling_dnf_rate"].fillna(0.05).values

    combined_dnf_rate = np.clip((driver_dnf_rate + team_dnf_rate) / 2, 0.01, 0.40)

    dnf_matrix = rng.random((n_sims, len(race_df))) < combined_dnf_rate[np.newaxis, :]
    return dnf_matrix


def simulate_safety_car(race_df, n_sims):
    sc_prob  = float(race_df["sc_probability"].iloc[0])
    vsc_prob = float(race_df["vsc_probability"].iloc[0])

    sc_occurs  = rng.random(n_sims) < sc_prob
    vsc_occurs = rng.random(n_sims) < vsc_prob

    return sc_occurs, vsc_occurs


def simulate_rainfall(race_df, n_sims):
    base_rainfall = float(race_df["rainfall"].iloc[0])

    if base_rainfall == 1:
        rain_prob = 0.85
    else:
        rain_prob = 0.10

    rain_occurs = rng.random(n_sims) < rain_prob
    return rain_occurs


def apply_sc_effect(scores, sc_occurs, vsc_occurs, n_drivers):
    sc_noise  = rng.normal(0, 1.5, (N_SIMULATIONS, n_drivers))
    vsc_noise = rng.normal(0, 0.8, (N_SIMULATIONS, n_drivers))

    scores = scores.copy()
    scores[sc_occurs]  += sc_noise[sc_occurs]
    scores[vsc_occurs] += vsc_noise[vsc_occurs]
    return scores


def apply_rain_effect(scores, rain_occurs, race_df, n_drivers):
    scores = scores.copy()

    rain_sims = np.where(rain_occurs)[0]
    if len(rain_sims) == 0:
        return scores

    grid_positions = race_df["grid_position"].fillna(10).values
    normalized_grid = (grid_positions - grid_positions.mean()) / (grid_positions.std() + 1e-6)

    rain_boost = rng.normal(0, 2.5, (len(rain_sims), n_drivers))
    rain_boost -= normalized_grid[np.newaxis, :] * 0.5

    scores[rain_sims] += rain_boost
    return scores


def apply_dnf_effect(scores, dnf_matrix):
    scores = scores.copy()
    large_penalty = -1000.0
    scores[dnf_matrix] = large_penalty
    return scores


def scores_to_positions(scores, dnf_matrix):
    n_sims, n_drivers = scores.shape
    positions = np.zeros_like(scores, dtype=int)

    for i in range(n_sims):
        sim_scores  = scores[i]
        sim_dnfs    = dnf_matrix[i]
        
        finisher_idx = np.where(~sim_dnfs)[0]
        dnf_idx      = np.where(sim_dnfs)[0]

        if len(finisher_idx) > 0:
            finisher_scores  = sim_scores[finisher_idx]
            finisher_order   = finisher_idx[np.argsort(-finisher_scores)]
            for pos, driver_i in enumerate(finisher_order):
                positions[i, driver_i] = pos + 1

        start_dnf_pos = len(finisher_idx) + 1
        for pos, driver_i in enumerate(dnf_idx):
            positions[i, driver_i] = start_dnf_pos + pos

    return positions


def compute_position_distribution(positions_matrix, n_drivers):
    n_sims = positions_matrix.shape[0]
    dist = np.zeros((n_drivers, n_drivers))

    for driver_i in range(n_drivers):
        for pos in range(1, n_drivers + 1):
            dist[driver_i, pos - 1] = np.sum(positions_matrix[:, driver_i] == pos) / n_sims

    return dist


def compute_summary_stats(positions_matrix, race_df):
    drivers = race_df["abbreviation"].values
    n_drivers = len(drivers)

    mean_pos   = positions_matrix.mean(axis=0)
    median_pos = np.median(positions_matrix, axis=0)
    std_pos    = positions_matrix.std(axis=0)

    win_prob    = (positions_matrix == 1).mean(axis=0)
    podium_prob = (positions_matrix <= 3).mean(axis=0)
    points_prob = (positions_matrix <= 10).mean(axis=0)
    dnf_prob    = (positions_matrix >= n_drivers - 2).mean(axis=0)

    summary = pd.DataFrame({
        "abbreviation":   drivers,
        "full_name":      race_df["full_name"].values,
        "team_name":      race_df["team_name"].values,
        "grid_position":  race_df["grid_position"].values,
        "mean_position":  np.round(mean_pos, 2),
        "median_position": np.round(median_pos, 2),
        "std_position":   np.round(std_pos, 2),
        "win_probability":    np.round(win_prob, 4),
        "podium_probability": np.round(podium_prob, 4),
        "points_probability": np.round(points_prob, 4),
        "dnf_probability":    np.round(dnf_prob, 4),
        "predicted_position": pd.array(mean_pos).argsort().argsort() + 1,
    })

    summary = summary.sort_values("mean_position").reset_index(drop=True)
    summary["predicted_position"] = range(1, len(summary) + 1)

    return summary


def compute_position_probability_table(positions_matrix, race_df):
    drivers   = race_df["abbreviation"].values
    n_drivers = len(drivers)
    dist      = compute_position_distribution(positions_matrix, n_drivers)

    cols = [f"P{i}" for i in range(1, n_drivers + 1)]
    df   = pd.DataFrame(dist, index=drivers, columns=cols)
    return df


def run_simulation(race_df, model, feature_cols):
    n_drivers = len(race_df)

    base_scores = get_base_scores(model, race_df, feature_cols)
    score_matrix = np.tile(base_scores, (N_SIMULATIONS, 1))

    lap_noise = rng.normal(0, 0.3, (N_SIMULATIONS, n_drivers))
    score_matrix += lap_noise

    dnf_matrix           = simulate_dnf(race_df, N_SIMULATIONS)
    sc_occurs, vsc_occurs = simulate_safety_car(race_df, N_SIMULATIONS)
    rain_occurs          = simulate_rainfall(race_df, N_SIMULATIONS)

    score_matrix = apply_sc_effect(score_matrix, sc_occurs, vsc_occurs, n_drivers)
    score_matrix = apply_rain_effect(score_matrix, rain_occurs, race_df, n_drivers)
    score_matrix = apply_dnf_effect(score_matrix, dnf_matrix)

    positions_matrix = scores_to_positions(score_matrix, dnf_matrix)

    summary   = compute_summary_stats(positions_matrix, race_df)
    prob_table = compute_position_probability_table(positions_matrix, race_df)

    return summary, prob_table, positions_matrix


def print_summary(summary, event_name, n_sims):
    print(f"\n{'=' * 60}")
    print(f"  {event_name}")
    print(f"  {n_sims:,} simulations")
    print(f"{'=' * 60}")
    print(f"  {'Pos':<5} {'Driver':<8} {'Team':<25} {'Grid':<6} {'Win%':<8} {'Podium%':<10} {'Points%'}")
    print(f"  {'-'*75}")

    for _, row in summary.iterrows():
        print(
            f"  {int(row['predicted_position']):<5}"
            f" {row['abbreviation']:<8}"
            f" {row['team_name']:<25}"
            f" {int(row['grid_position']) if pd.notna(row['grid_position']) else '?':<6}"
            f" {row['win_probability']:.1%}   "
            f" {row['podium_probability']:.1%}      "
            f" {row['points_probability']:.1%}"
        )


def simulate_race(year, round_num, processed_data_path=None):
    if processed_data_path is None:
        processed_data_path = os.path.join(INPUT_DIR, "processed_data.csv")

    df = pd.read_csv(processed_data_path)
    race_df = df[(df["year"] == year) & (df["round"] == round_num)].copy()

    if race_df.empty:
        raise ValueError(f"No data found for year={year} round={round_num}")

    model, feature_cols = load_model()

    event_name = race_df["event_name"].iloc[0]
    print(f"\nSimulating {event_name} ({year} R{round_num})  —  {N_SIMULATIONS:,} runs...")

    summary, prob_table, positions_matrix = run_simulation(race_df, model, feature_cols)

    print_summary(summary, event_name, N_SIMULATIONS)

    summary_path = os.path.join(OUTPUT_DIR, f"sim_{year}_R{round_num}_summary.csv")
    prob_path    = os.path.join(OUTPUT_DIR, f"sim_{year}_R{round_num}_probabilities.csv")

    summary.to_csv(summary_path, index=False)
    prob_table.to_csv(prob_path)

    print(f"\n  Summary saved      : {summary_path}")
    print(f"  Probability table  : {prob_path}")

    return summary, prob_table


if __name__ == "__main__":
    print("=" * 60)
    print("F1 Race Simulation  |  Monte Carlo")
    print("=" * 60)

    df = pd.read_csv(os.path.join(INPUT_DIR, "processed_data.csv"))

    last_race = df.sort_values(["year", "round"]).iloc[-1]
    year      = int(last_race["year"])
    round_num = int(last_race["round"])

    summary, prob_table = simulate_race(year, round_num)

    print("\n" + "=" * 60)
    print("✅  Simulation complete.")
    print("=" * 60)
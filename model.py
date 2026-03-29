import os
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

INPUT_DIR = "./data"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "grid_position",
    "gap_to_pole_s",
    "best_q_time_s",
    "start_compound_encoded",
    "tyre_life_start",
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
    "rainfall",
    "circuit_encoded",
    "team_encoded",
    "driver_encoded",
]

TARGET_COL = "finish_position"
GROUP_COL  = "race_id"


def load_data():
    df = pd.read_csv(os.path.join(INPUT_DIR, "processed_data.csv"))
    df["race_id"] = df["year"].astype(str) + "_" + df["round"].astype(str)
    return df


def split_by_era(df):
    train = df[df["year"] <= 2021].copy()
    val   = df[(df["year"] >= 2022) & (df["year"] <= 2023)].copy()
    test  = df[df["year"] == 2024].copy()
    return train, val, test


def prepare_xy(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET_COL].copy()

    X = X.fillna(X.median())
    y = y.fillna(20)

    return X, y


def get_group_sizes(df):
    return df.groupby(GROUP_COL, sort=False).size().values


def train_model(X_train, y_train, groups_train, X_val, y_val, groups_val):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)

    dtrain.set_group(groups_train)
    dval.set_group(groups_val)

    params = {
        "objective":        "rank:pairwise",
        "eval_metric":      "ndcg",
        "eta":              0.05,
        "max_depth":        5,
        "min_child_weight": 3,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "lambda":           1.5,
        "alpha":            0.5,
        "seed":             42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )

    return model


def predict_race_order(model, df):
    X, _ = prepare_xy(df)
    dmat  = xgb.DMatrix(X)
    scores = model.predict(dmat)

    results = df[["race_id", "year", "round", "event_name",
                  "abbreviation", "full_name", "team_name",
                  "grid_position", TARGET_COL]].copy()
    results["predicted_score"] = scores
    results["predicted_position"] = (
        results.groupby("race_id")["predicted_score"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    return results


def evaluate(results):
    correlations = []

    for race_id, group in results.groupby("race_id"):
        if len(group) < 5:
            continue
        corr, _ = spearmanr(group["finish_position"], group["predicted_position"])
        correlations.append(corr)

    mean_corr   = np.mean(correlations)
    median_corr = np.median(correlations)
    min_corr    = np.min(correlations)
    max_corr    = np.max(correlations)

    print(f"\n  Spearman Rank Correlation across {len(correlations)} races:")
    print(f"    Mean   : {mean_corr:.4f}")
    print(f"    Median : {median_corr:.4f}")
    print(f"    Min    : {min_corr:.4f}")
    print(f"    Max    : {max_corr:.4f}")

    top3_correct = []
    winner_correct = []

    for race_id, group in results.groupby("race_id"):
        actual_top3    = set(group[group["finish_position"] <= 3]["abbreviation"])
        predicted_top3 = set(group[group["predicted_position"] <= 3]["abbreviation"])
        actual_winner    = set(group[group["finish_position"] == 1]["abbreviation"])
        predicted_winner = set(group[group["predicted_position"] == 1]["abbreviation"])

        top3_correct.append(len(actual_top3 & predicted_top3) / 3)
        winner_correct.append(int(actual_winner == predicted_winner))

    print(f"\n  Winner accuracy      : {np.mean(winner_correct):.2%}")
    print(f"  Avg podium overlap   : {np.mean(top3_correct):.2%}")

    return mean_corr


def cross_validate(df, n_splits=5):
    print("\nRunning cross-validation...")

    le = LabelEncoder()
    df["race_id_encoded"] = le.fit_transform(df["race_id"])

    gkf = GroupKFold(n_splits=n_splits)
    X, y = prepare_xy(df)
    groups = df["race_id_encoded"]

    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        g_tr = df.iloc[train_idx].groupby("race_id", sort=False).size().values
        g_va = df.iloc[val_idx].groupby("race_id", sort=False).size().values

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_va, label=y_va)
        dtrain.set_group(g_tr)
        dval.set_group(g_va)

        params = {
            "objective":        "rank:pairwise",
            "eval_metric":      "ndcg",
            "eta":              0.05,
            "max_depth":        5,
            "min_child_weight": 3,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "lambda":           1.5,
            "alpha":            0.5,
            "seed":             42,
            "verbosity":        0,
        }

        m = xgb.train(params, dtrain, num_boost_round=500, verbose_eval=False)

        scores = m.predict(dval)
        val_df = df.iloc[val_idx].copy()
        val_df["predicted_score"] = scores
        val_df["predicted_position"] = (
            val_df.groupby("race_id")["predicted_score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        fold_corrs = []
        for race_id, group in val_df.groupby("race_id"):
            if len(group) < 5:
                continue
            corr, _ = spearmanr(group["finish_position"], group["predicted_position"])
            fold_corrs.append(corr)

        fold_score = np.mean(fold_corrs)
        cv_scores.append(fold_score)
        print(f"  Fold {fold + 1}: Spearman = {fold_score:.4f}")

    print(f"\n  CV Mean Spearman: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    return cv_scores


def save_model(model, feature_cols):
    model_path   = os.path.join(MODEL_DIR, "xgb_f1_ranker.ubj")
    feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
    model.save_model(model_path)
    joblib.dump(feature_cols, feature_path)
    print(f"\n  Model saved   : {model_path}")
    print(f"  Features saved: {feature_path}")


def print_feature_importance(model, feature_cols):
    importance = model.get_score(importance_type="gain")
    imp_df = (
        pd.DataFrame(importance.items(), columns=["feature", "gain"])
        .sort_values("gain", ascending=False)
        .head(15)
    )
    print("\n  Top 15 features by gain:")
    for _, row in imp_df.iterrows():
        bar = "█" * int(row["gain"] / imp_df["gain"].max() * 30)
        print(f"    {row['feature']:<35} {bar}")


if __name__ == "__main__":
    print("=" * 60)
    print("F1 Race Prediction  |  XGBoost Ranking Model")
    print("=" * 60)

    print("\nLoading processed data...")
    df = load_data()
    print(f"  {len(df)} rows  |  {df['race_id'].nunique()} races  |  {df['year'].nunique()} seasons")

    print("\nSplitting into train / val / test by era...")
    train_df, val_df, test_df = split_by_era(df)
    print(f"  Train : {len(train_df)} rows ({train_df['year'].min()}–{train_df['year'].max()})")
    print(f"  Val   : {len(val_df)} rows ({val_df['year'].min()}–{val_df['year'].max()})")
    print(f"  Test  : {len(test_df)} rows ({test_df['year'].min()}–{test_df['year'].max()})")

    X_train, y_train = prepare_xy(train_df)
    X_val,   y_val   = prepare_xy(val_df)

    groups_train = get_group_sizes(train_df)
    groups_val   = get_group_sizes(val_df)

    print("\nTraining model...")
    model = train_model(X_train, y_train, groups_train, X_val, y_val, groups_val)

    print("\n--- Validation Set Performance ---")
    val_results = predict_race_order(model, val_df)
    evaluate(val_results)

    print("\n--- Test Set Performance (2024) ---")
    test_results = predict_race_order(model, test_df)
    evaluate(test_results)

    print("\n--- Cross Validation (full dataset) ---")
    cross_validate(df)

    print("\n--- Feature Importance ---")
    available_features = [c for c in FEATURE_COLS if c in X_train.columns]
    print_feature_importance(model, available_features)

    print("\nSaving model...")
    save_model(model, available_features)

    results_path = os.path.join(INPUT_DIR, "test_predictions.csv")
    test_results.to_csv(results_path, index=False)
    print(f"\n  Test predictions saved: {results_path}")

    print("\n" + "=" * 60)
    print("✅  Model training complete.")
    print("=" * 60)

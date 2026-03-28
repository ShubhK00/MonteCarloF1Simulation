import os
import warnings
import pandas as pd
import fastf1
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Config

YEARS = list(range(2018, 2025))          
CACHE_DIR = "./ff1_cache"                
OUTPUT_DIR = "./data"         

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE_DIR)

# Helpers

def safe_seconds(td):
    try:
        return td.total_seconds() if pd.notna(td) else None
    except Exception:
        return None


def get_schedule(year):
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule[schedule["RoundNumber"] > 0]


# CSV1 - Race Results

def pull_race_results(years):

    rows = []

    for year in years:
        schedule = get_schedule(year)

        for _, event in tqdm(schedule.iterrows(),
                             total=len(schedule),
                             desc=f"Race results {year}"):
            round_num = event["RoundNumber"]
            event_name = event["EventName"]
            circuit = event["Location"]
            circuit_type = event.get("EventFormat", "conventional")
            country = event["Country"]

            try:
                session = fastf1.get_session(year, round_num, "R")
                session.load(laps=False, telemetry=False,
                             weather=False, messages=False)

                for _, driver in session.results.iterrows():
                    rows.append({
                        "year":             year,
                        "round":            round_num,
                        "event_name":       event_name,
                        "circuit":          circuit,
                        "country":          country,
                        "circuit_type":     circuit_type,
                        "driver_number":    driver["DriverNumber"],
                        "abbreviation":     driver["Abbreviation"],
                        "full_name":        driver["FullName"],
                        "team_name":        driver["TeamName"],
                        "grid_position":    driver["GridPosition"],
                        "finish_position":  driver["Position"],
                        "classified_pos":   driver["ClassifiedPosition"],
                        "status":           driver["Status"],
                        "points":           driver["Points"],
                        "total_laps":       driver["Laps"],
                        "dnf": int(str(driver["Status"]).upper() not in
                                   ["FINISHED", "+1 LAP", "+2 LAPS",
                                    "+3 LAPS", "+4 LAPS", "+5 LAPS"]),
                        "race_time_s":      safe_seconds(driver.get("Time")),
                    })

            except Exception as e:
                print(f"  Warning  {year} R{round_num} race results failed: {e}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "race_results.csv")
    df.to_csv(path, index=False)
    print(f"\n✅  race_results.csv  →  {len(df)} driver-race rows")
    return df


# CSV2 - Qualifying Results

def pull_qualifying_results(years):

    rows = []

    for year in years:
        schedule = get_schedule(year)

        for _, event in tqdm(schedule.iterrows(),
                             total=len(schedule),
                             desc=f"Qualifying {year}"):
            round_num = event["RoundNumber"]
            event_name = event["EventName"]

            try:
                session = fastf1.get_session(year, round_num, "Q")
                session.load(laps=False, telemetry=False,
                             weather=False, messages=False)

                results = session.results

                pole_row = results[results["Position"] == 1]
                if pole_row.empty:
                    continue

                def best_q_time(row):
                    for col in ["Q3", "Q2", "Q1"]:
                        t = safe_seconds(row.get(col))
                        if t:
                            return t
                    return None

                pole_time = best_q_time(pole_row.iloc[0])

                for _, driver in results.iterrows():
                    best_time = best_q_time(driver)
                    gap_to_pole = (
                        round(best_time - pole_time, 4)
                        if best_time and pole_time else None
                    )

                    rows.append({
                        "year":          year,
                        "round":         round_num,
                        "event_name":    event_name,
                        "driver_number": driver["DriverNumber"],
                        "abbreviation":  driver["Abbreviation"],
                        "team_name":     driver["TeamName"],
                        "grid_position": driver["Position"],
                        "q1_time_s":     safe_seconds(driver.get("Q1")),
                        "q2_time_s":     safe_seconds(driver.get("Q2")),
                        "q3_time_s":     safe_seconds(driver.get("Q3")),
                        "best_q_time_s": best_time,
                        "gap_to_pole_s": gap_to_pole,
                    })

            except Exception as e:
                print(f"  Warning  {year} R{round_num} qualifying failed: {e}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "qualifying_results.csv")
    df.to_csv(path, index=False)
    print(f"\n✅  qualifying_results.csv  →  {len(df)} driver-race rows")
    return df


# CSV3 - Pre-Race Weather

def pull_weather_snapshots(years):

    rows = []

    for year in years:
        schedule = get_schedule(year)

        for _, event in tqdm(schedule.iterrows(),
                             total=len(schedule),
                             desc=f"Weather {year}"):
            round_num = event["RoundNumber"]
            event_name = event["EventName"]

            try:
                session = fastf1.get_session(year, round_num, "R")
                session.load(laps=False, telemetry=False,
                             weather=True, messages=False)

                w = session.weather_data
                if w is None or w.empty:
                    continue

                snap = w.iloc[0]

                rows.append({
                    "year":            year,
                    "round":           round_num,
                    "event_name":      event_name,
                    "air_temp_c":      snap.get("AirTemp"),
                    "track_temp_c":    snap.get("TrackTemp"),
                    "humidity_pct":    snap.get("Humidity"),
                    "pressure_mbar":   snap.get("Pressure"),
                    "wind_speed_ms":   snap.get("WindSpeed"),
                    "wind_dir_deg":    snap.get("WindDirection"),
                    "rainfall":        int(bool(snap.get("Rainfall", 0))),
                })

            except Exception as e:
                print(f"  Warning  {year} R{round_num} weather failed: {e}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "weather_snapshots.csv")
    df.to_csv(path, index=False)
    print(f"\n✅  weather_snapshots.csv  →  {len(df)} race rows")
    return df


# CSV4 - Race Control Events

def pull_race_control(years):

    rows = []

    for year in years:
        schedule = get_schedule(year)

        for _, event in tqdm(schedule.iterrows(),
                             total=len(schedule),
                             desc=f"Race control {year}"):
            round_num = event["RoundNumber"]
            event_name = event["EventName"]
            circuit = event["Location"]

            try:
                session = fastf1.get_session(year, round_num, "R")
                session.load(laps=False, telemetry=False,
                             weather=False, messages=True)

                rc = session.race_control_messages
                if rc is None or rc.empty:
                    continue

                for _, msg in rc.iterrows():
                    rows.append({
                        "year":        year,
                        "round":       round_num,
                        "event_name":  event_name,
                        "circuit":     circuit,
                        "time":        str(msg.get("Time")),
                        "lap_number":  msg.get("Lap"),
                        "category":    msg.get("Category"),
                        "message":     msg.get("Message"),
                        "flag":        msg.get("Flag"),
                        "scope":       msg.get("Scope"),
                        "status":      msg.get("Status"),
                    })

            except Exception as e:
                print(f"  Warning  {year} R{round_num} race control failed: {e}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "race_control.csv")
    df.to_csv(path, index=False)
    print(f"\n✅  race_control.csv  →  {len(df)} messages")
    return df


# CSV5 - Starting Tyre Compounds

def pull_starting_stints(years):

    rows = []

    for year in years:
        schedule = get_schedule(year)

        for _, event in tqdm(schedule.iterrows(),
                             total=len(schedule),
                             desc=f"Starting stints {year}"):
            round_num = event["RoundNumber"]
            event_name = event["EventName"]

            try:
                session = fastf1.get_session(year, round_num, "R")
                session.load(laps=True, telemetry=False,
                             weather=False, messages=False)

                laps = session.laps

                stint1 = (
                    laps[laps["LapNumber"] == 1][
                        ["DriverNumber", "Compound", "TyreLife", "FreshTyre"]
                    ]
                    .drop_duplicates(subset="DriverNumber")
                )

                for _, row in stint1.iterrows():
                    rows.append({
                        "year":            year,
                        "round":           round_num,
                        "event_name":      event_name,
                        "driver_number":   row["DriverNumber"],
                        "start_compound":  row.get("Compound"),
                        "tyre_life_start": row.get("TyreLife"),
                        "fresh_tyre":      row.get("FreshTyre"),
                    })

            except Exception as e:
                print(f"  Warning  {year} R{round_num} stints failed: {e}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "starting_stints.csv")
    df.to_csv(path, index=False)
    print(f"\n✅  starting_stints.csv  →  {len(df)} driver-race rows")
    return df


#Main

if __name__ == "__main__":
    print("=" * 60)
    print("F1 Data Ingestion  |  2018–2024  |  FastF1")
    print("=" * 60)
    print(f"Cache dir : {os.path.abspath(CACHE_DIR)}")
    print(f"Output dir: {os.path.abspath(OUTPUT_DIR)}")
    print()
    print("⏳  This will take 30–90 minutes on first run.")
    print("    Subsequent runs use the local cache and are fast.\n")

    #race_results      = pull_race_results(YEARS)
    qualifying        = pull_qualifying_results(YEARS)
    #weather           = pull_weather_snapshots(YEARS)
    #race_control      = pull_race_control(YEARS)
    #starting_stints   = pull_starting_stints(YEARS)

    print("\n" + "=" * 60)
    print("✅  All data pulled successfully.")
    print(f"    Files saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)

import random
import numpy as np
import plotly.graph_objects as go
import sys

# === DATA ===
drivers = ["Oscar Piastri", "Lando Norris", "Charles Leclerc", "Lewis Hamilton", "George Russell", 
           "Kimi Antonelli", "Max Verstappen", "Yuki Tsunoda", "Alexander Albon", "Carlos Sainz", 
           "Nico Hulkenberg", "Gabriel Bortoleto", "Liam Lawson", "Isack Hadjar", "Lance Stroll", 
           "Fernando Alonso", "Esteban Ocon", "Oliver Bearman", "Pierre Gasly", "Franco Colapinto"]

driver_scores = [266, 250, 139, 109, 157, 63, 185, 10, 54, 16, 37, 6, 16, 22, 20, 16, 27, 8, 20, 0]
driver_wins = [5, 4, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
avg_positions = [2.2, 3.3, 6.0, 7.2, 5.1, 10.4, 3.5, 11.0, 9.1, 12.0, 8.5, 12.8, 13.2, 14.5, 11.5, 16.0, 9.5, 13.8, 14.0, 15.0]

races_left = 11
total_races = 24
points_per_position = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0]*10

# === SKILL SCORES ===
def skillscore(score, wincount, avg_finish):
    races_run = total_races - races_left
    win_factor = wincount / races_run if races_run > 0 else 0
    point_factor = score / (25 * races_run) if races_run > 0 else 0
    avg_finish_factor = 1 - (avg_finish - 1) / 19
    return 0.2 * win_factor + 0.5 * point_factor + 0.3 * avg_finish_factor

driver_skillscore = [
    skillscore(driver_scores[i], driver_wins[i], avg_positions[i]) 
    for i in range(len(drivers))
]

total_skill = sum(driver_skillscore)
weights = [s / total_skill for s in driver_skillscore]

# === SIMULATION FUNCTIONS ===
def weighted_race_simulation(drivers, weights):
    remaining_drivers = drivers.copy()
    remaining_weights = weights.copy()
    finishing_order = []
    while remaining_drivers:
        chosen = random.choices(remaining_drivers, weights=remaining_weights, k=1)[0]
        finishing_order.append(chosen)
        idx = remaining_drivers.index(chosen)
        remaining_drivers.pop(idx)
        remaining_weights.pop(idx)
    return finishing_order

def monte_carlo_finishing_positions(drivers, weights, num_simulations=100000):
    counts = {driver: [0]*len(drivers) for driver in drivers}
    for _ in range(num_simulations):
        order = weighted_race_simulation(drivers, weights)
        for pos, driver in enumerate(order):
            counts[driver][pos] += 1
    probabilities = {
        driver: [count/sum(pos_counts) for count in pos_counts]
        for driver, pos_counts in counts.items()
    }
    return probabilities

def simulate_season_projection(drivers, weights, current_scores, races_left, num_simulations=10000):
    total_points_samples = {driver: [] for driver in drivers}
    for _ in range(num_simulations):
        sim_scores = current_scores.copy()
        for _ in range(races_left):
            order = weighted_race_simulation(drivers, weights)
            for pos, driver in enumerate(order):
                sim_scores[drivers.index(driver)] += points_per_position[pos]
        for i, driver in enumerate(drivers):
            total_points_samples[driver].append(sim_scores[i])
    return total_points_samples

# === VISUALIZATION FUNCTIONS ===
def show_position_plot(driver_name, position_results):
    positions = list(range(1, len(drivers)+1))
    probs = position_results[driver_name]
    fig = go.Figure(data=[go.Bar(
        x=positions,
        y=probs,
        text=[f"{p*100:.2f}%" for p in probs],
        hoverinfo='text+x',
        marker_color='royalblue'
    )])
    fig.update_layout(
        title=f"Finishing Position Probabilities for {driver_name}",
        xaxis_title="Finishing Position",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        xaxis=dict(dtick=1)
    )
    fig.show()

def show_season_projection_plot(driver_name, season_results):
    samples = season_results[driver_name]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples,
        nbinsx=50,
        name=f"{driver_name} Season Points",
        marker_color='crimson',
        opacity=0.75,
    ))
    fig.update_layout(
        title=f"Projected Total Points Distribution for {driver_name}",
        xaxis_title="Total Points at Season End",
        yaxis_title="Frequency",
        bargap=0.1
    )
    fig.show()

# === MAIN RUN ===
print("Running simulations... please wait.")
position_results = monte_carlo_finishing_positions(drivers, weights)
season_results = simulate_season_projection(drivers, weights, driver_scores.copy(), races_left)

championship_wins = {driver: 0 for driver in drivers}
num_simulations = len(next(iter(season_results.values())))
for i in range(num_simulations):
    scores = {driver: season_results[driver][i] for driver in drivers}
    max_points = max(scores.values())
    winners = [d for d, pts in scores.items() if pts == max_points]
    for w in winners:
        championship_wins[w] += 1 / len(winners)
championship_probs = {d: championship_wins[d]/num_simulations for d in drivers}

print("\nTop 5 Drivers by Expected Final Points:")
expected_points = {d: np.mean(season_results[d]) for d in drivers}
top5 = sorted(expected_points.items(), key=lambda x: x[1], reverse=True)[:5]
for driver, points in top5:
    print(f"{driver}: {points:.1f} pts, Win Chance: {championship_probs[driver]*100:.2f}%")

# === USER INTERFACE ===
while True:
    mode = input("\nType 'position' to see race finish probabilities or 'season' for season projection (or 'exit'): ").strip().lower()
    if mode == "exit":
        print("Exiting.")
        break
    elif mode not in ["position", "season"]:
        print("Invalid mode. Please type 'position' or 'season'.")
        continue

    driver_name = input(f"Enter a driver name (or 'back' to return):\nAvailable: {', '.join(drivers)}\n> ").strip()
    if driver_name.lower() == 'back':
        continue
    if driver_name not in drivers:
        print("Invalid driver name.")
        continue

    if mode == 'position':
        show_position_plot(driver_name, position_results)
    elif mode == 'season':
        show_season_projection_plot(driver_name, season_results)
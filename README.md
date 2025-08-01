# MonteCarloF1Simulation
Uses Monte-Carlo Distribution and real F1 statistics to generate predictions for the rest of the season.

Features:
- Simulate race finishing order probabilities
- Simulate remaining season outcomes
- Visualize interactive charts using Plotly
- Hover to see percentages or point totals
- See expected final points and championship odds
- Switch between position and season views dynamically


🧠 How It Works
1. Each driver is assigned a skill score based on:
  - Current Points
  - Win Count
- Average Finishing Position
2. Then:
- Monte Carlo simulates 100,000+ single races using weighted probabilities.
- Monte Carlo simulates 10,000 full seasons using the F1 points system.
- Results are displayed as interactive plots.


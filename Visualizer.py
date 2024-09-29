import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "play_tennis.csv"
data = pd.read_csv(file_path)

# Create a crosstab to count occurrences of 'play' for each 'wind'
wind_play = pd.crosstab(data["wind"], data["play"])

# Plot a bar chart for each category of 'wind'
wind_play.plot(kind="bar", color=["red", "green"])

# Add labels and title
plt.title("Bar Chart of Tennis Play Decision by Weather wind")
plt.xlabel("wind")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.legend(title="Play", loc="upper right")

# Show the plot
plt.show()

# analyse data from 0.0001.csv and 0.0002.csv and plot graphs
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
data = pd.read_csv("0.0001.csv")
data2 = pd.read_csv("0.0002.csv")

# Plot the data against serial number
plt.scatter(range(len(data["m"])), data["m"], c="black", marker=".")
plt.scatter(range(len(data["b"])), data["b"], c="red", marker=".")
plt.scatter(range(len(data["loss"])), data["loss"], c="blue", marker=".")
plt.show()

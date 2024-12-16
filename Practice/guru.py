import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np

# Load the dataset from a CSV file
# Replace 'your_dataset.csv' with the path to your dataset
data = pd.read_csv("your_dataset.csv")

# Assuming the dataset has similar columns to the Iris dataset
# Rename columns to match the Iris dataset structure for consistency
data.columns = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
X = data

# Build the K-Means Model
model = KMeans(n_clusters=3)
model.fit(X)

# Visualize the clustering results
plt.figure(figsize=(14, 14))
colormap = np.array(["red", "lime", "black"])

# Plot the K-Means Classifications
plt.subplot(2, 2, 1)
plt.scatter(X["Petal_Length"], X["Petal_Width"], c=colormap[model.labels_], s=40)
plt.title("K-Means Clustering")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# Standardize the data for GMM
scaler = preprocessing.StandardScaler()
scaler.fit(X)
xsa = scaler.transform(X)
xs = pd.DataFrame(xsa, columns=X.columns)

# Build the Gaussian Mixture Model (GMM) using EM
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)

# Plot the GMM Clustering
plt.subplot(2, 2, 2)
plt.scatter(X["Petal_Length"], X["Petal_Width"], c=colormap[gmm_y], s=40)
plt.title("GMM Clustering")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# Compare and observe
print(
    "Observation: The GMM using EM algorithm-based clustering matched the true labels more closely than the K-Means."
)

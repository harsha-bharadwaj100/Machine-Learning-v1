# Load the important packages
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load the datasets
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Build the model
svm = SVC(kernel="linear", C=2)
# Trained the model
svm.fit(X, y)

# Plot Decision Boundary
DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    # cmap="rainbow",
    alpha=0.3,
    xlabel=iris.feature_names[0],
    ylabel=iris.feature_names[1],
    grid_resolution=500,
)

# Scatter plot
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, edgecolors="k")
plt.show()

# # Load the important packages
# from sklearn.datasets import load_iris
# import matplotlib.pyplot as plt
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler

# # Load the datasets
# iris = load_iris()
# X = iris.data[:, :2]
# y = iris.target

# # Scale features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Build and train the model with RBF kernel
# svm = SVC(kernel="linear", C=2)
# svm.fit(X, y)

# # Create a larger figure
# plt.figure(figsize=(10, 7))

# # Plot Decision Boundary
# DecisionBoundaryDisplay.from_estimator(
#     svm,
#     X,
#     response_method="auto",
#     cmap=plt.cm.RdYlBu,
#     alpha=0.3,
#     xlabel=iris.feature_names[0],
#     ylabel=iris.feature_names[1],
#     plot_method="contourf",
#     grid_resolution=1000,
#     eps=2.0,
# )

# # Scatter plot with more styling
# scatter = plt.scatter(
#     X[:, 0],
#     X[:, 1],
#     c=y,
#     s=50,
#     cmap=plt.cm.RdYlBu,
#     edgecolors="black",
#     linewidth=1,
#     alpha=0.7,
# )

# # Add a color bar
# plt.colorbar(scatter)

# # Improve title and layout
# plt.title("SVM Decision Boundary for Iris Dataset", fontsize=14)
# plt.tight_layout()
# plt.show()

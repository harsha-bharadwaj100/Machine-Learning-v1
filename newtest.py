import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Set the seaborn style
sns.set_style("whitegrid")
sns.set_palette("deep")

# Load the data
data = pd.read_csv("play_tennis.csv")

# Prepare the features and target
X = data.drop(["day", "play"], axis=1)
y = data["play"]

# Encode categorical variables
le = LabelEncoder()
for column in X.columns:
    X[column] = le.fit_transform(X[column])

y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": clf.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Seaborn visualization for feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="feature", y="importance", data=feature_importance)
plt.title("Feature Importance", fontsize=16)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Importance", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("feature_importance_seaborn.png")
print("\nFeature importance visualization saved as 'feature_importance_seaborn.png'")

# Decision Tree visualization using matplotlib with seaborn styling
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=10,
)
plt.title("Decision Tree for Tennis Play Prediction", fontsize=20)
plt.tight_layout()
plt.savefig("tennis_decision_tree_seaborn_styled.png", dpi=300, bbox_inches="tight")
print(
    "\nDecision tree visualization saved as 'tennis_decision_tree_seaborn_styled.png'"
)


# Function to plot feature importances with seaborn
def plot_feature_importances(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices])
    plt.title("Feature Importances in Decision Tree", fontsize=16)
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Features", fontsize=12)
    plt.tight_layout()
    plt.savefig("feature_importances_detailed_seaborn.png")
    print(
        "\nDetailed feature importances visualization saved as 'feature_importances_detailed_seaborn.png'"
    )


# Plot detailed feature importances
plot_feature_importances(clf.feature_importances_, X.columns)

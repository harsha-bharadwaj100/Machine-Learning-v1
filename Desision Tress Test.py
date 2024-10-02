import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

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
    X, y, test_size=0.3, random_state=42
)

# Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# Visualize the decision tree
plt.figure(figsize=(50, 50))
plot_tree(
    clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, rounded=True
)
plt.savefig("tennis_decision_tree.png")
print("\nDecision tree visualization saved as 'tennis_decision_tree.png'")

# Feature importance
feature_importance = pd.DataFrame(
    {"feature": X.columns, "importance": clf.feature_importances_}
)
feature_importance = feature_importance.sort_values("importance", ascending=False)
print("\nFeature Importance:")
print(feature_importance)

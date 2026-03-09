from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

# Load Breast Cancer Dataset
data = load_breast_cancer()

X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

print("Dataset Loaded Successfully")
print("Classes:", class_names)

# Convert dataset to DataFrame
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# Convert numeric target to text
df["target"] = df["target"].map({
    0: "malignant",
    1: "benign"
})

# Save dataset
df.to_csv("breast_cancer_dataset.csv", index=False)

print("Dataset saved as breast_cancer_dataset.csv")

# Create Random Forest Model
model = RandomForestClassifier(
    n_estimators=5,
    criterion="entropy",
    max_depth=3,
    random_state=42
)

# Train model
model.fit(X, y)

print("\nRandom Forest built using ENTROPY")

# Display all trees
for i, tree in enumerate(model.estimators_):

    plt.figure(figsize=(20,10))

    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True
    )

    plt.title(f"Decision Tree {i+1}")

    plt.show()

# Test prediction
new_sample = X[0].reshape(1, -1)

print("\nTree-wise Predictions:\n")

tree_predictions = []

for i, tree in enumerate(model.estimators_):

    pred = tree.predict(new_sample).astype(int)
    decoded = class_names[pred][0]

    tree_predictions.append(decoded)

    print(f"Tree {i+1} Prediction:", decoded)

# Majority Voting
final_vote = max(set(tree_predictions), key=tree_predictions.count)

print("\nFinal Prediction (Majority Voting):", final_vote)
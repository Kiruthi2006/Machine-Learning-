import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

# Map numeric target to class names
df["target"] = df["target"].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

# Save dataset
df.to_csv("iris.csv", index=False)

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train model
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X, y)

print("kiruthiga J-2303717710422703")

# Function to parse input values
def parse_input_values(raw: str):
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != X.shape[1]:
        raise ValueError(f"Expected {X.shape[1]} values, got {len(parts)}")
    return [float(p) for p in parts]

# Get user input
def get_user_input(args):

    if args.input:
        vals = parse_input_values(args.input)
        return vals

    if args.interactive:
        vals = []
        print("Enter feature values:")
        for col in X.columns:
            while True:
                try:
                    v = input(f"{col}: ")
                    vals.append(float(v))
                    break
                except ValueError:
                    print("Enter a valid number")
        return vals

    if sys.stdin.isatty():
        resp = input("No input provided. Enter values now? (y/N): ")
        if resp.lower().startswith('y'):
            vals = []
            for col in X.columns:
                while True:
                    try:
                        v = input(f"{col}: ")
                        vals.append(float(v))
                        break
                    except ValueError:
                        print("Enter a valid number")
            return vals

    return [5.1, 3.5, 1.4, 0.2]

# Main function
def main():

    parser = argparse.ArgumentParser(
        description="Decision Tree Classification on Iris Dataset"
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Comma separated values (example: 5.1,3.5,1.4,0.2)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter feature values manually"
    )

    args = parser.parse_args()

    sample = get_user_input(args)

    prediction = model.predict(
        pd.DataFrame([sample], columns=X.columns)
    )

    print("Predicted Class:", prediction[0])

if __name__ == "__main__":
    main()

# Plot decision tree
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=12
)

plt.savefig("iris_decision_tree.pdf", bbox_inches="tight")
plt.show()
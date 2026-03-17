import pandas as pd
import tkinter as tk
from tkinter import messagebox
import urllib.request
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Download dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
urllib.request.urlretrieve(url, "heart.csv")

columns = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal","target"
]

# ✅ USE SAME FILE
data = pd.read_csv("heart.csv", names=columns)

# ✅ CLEAN DATA
data = data.replace("?", pd.NA)
data = data.dropna()
data = data.apply(pd.to_numeric)

# Convert target
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Select features
data = data[['age','sex','cp','chol','thalach','target']]

# Convert to categories
data['age'] = pd.cut(data['age'], bins=3, labels=[0,1,2])
data['chol'] = pd.cut(data['chol'], bins=3, labels=[0,1,2])
data['thalach'] = pd.cut(data['thalach'], bins=3, labels=[0,1,2])

# Model
model = DiscreteBayesianNetwork([
    ('age','target'),
    ('sex','target'),
    ('cp','target'),
    ('chol','target'),
    ('thalach','target')
])

model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

# GUI
def predict():
    try:
        age = int(age_entry.get())
        sex = int(sex_entry.get())
        cp = int(cp_entry.get())
        chol = int(chol_entry.get())
        thalach = int(thalach_entry.get())

        result = inference.query(
            variables=['target'],
            evidence={
                'age': age,
                'sex': sex,
                'cp': cp,
                'chol': chol,
                'thalach': thalach
            }
        )

        prob = result.values[1]

        messagebox.showinfo(
            "Result",
            f"Heart Disease Probability: {prob:.2f}"
        )

    except:
        messagebox.showerror("Error", "Enter valid values")

root = tk.Tk()
root.title("Heart Disease Prediction")
root.geometry("400x420")

tk.Label(root, text="Heart Disease Prediction", font=("Arial",14)).pack(pady=10)

tk.Label(root, text="Age (0-2)").pack()
age_entry = tk.Entry(root)
age_entry.pack()

tk.Label(root, text="Sex (0/1)").pack()
sex_entry = tk.Entry(root)
sex_entry.pack()

tk.Label(root, text="Chest Pain (0-3)").pack()
cp_entry = tk.Entry(root)
cp_entry.pack()

tk.Label(root, text="Chol (0-2)").pack()
chol_entry = tk.Entry(root)
chol_entry.pack()

tk.Label(root, text="Thalach (0-2)").pack()
thalach_entry = tk.Entry(root)
thalach_entry.pack()

tk.Button(root, text="Predict", command=predict).pack(pady=20)

root.mainloop()
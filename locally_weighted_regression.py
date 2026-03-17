import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Kaggle dataset
data = pd.read_csv("Salary_Data.csv")

# Feature and Target
X = data['YearsExperience'].values
y = data['Salary'].values

# Add bias term
X = np.vstack([np.ones(len(X)), X]).T


# Locally Weighted Regression Function
def lwlr(test_point, X, y, tau):
    m = X.shape[0]
    weights = np.eye(m)

    for i in range(m):
        diff = test_point - X[i]
        weights[i, i] = np.exp(diff @ diff.T / (-2 * tau**2))

    theta = np.linalg.inv(X.T @ weights @ X) @ X.T @ weights @ y
    return test_point @ theta


# Predict for all points
def lwlr_test(X, y, tau):
    y_pred = []
    for point in X:
        y_pred.append(lwlr(point, X, y, tau))
    return np.array(y_pred)


tau = 0.5
y_pred = lwlr_test(X, y, tau)

# Sort for smooth graph
sort_index = X[:,1].argsort()

# Plot
plt.scatter(X[:,1], y, color='red', label='Original Data')
plt.plot(X[:,1][sort_index], y_pred[sort_index], color='blue', label='LWLR Fit')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Locally Weighted Linear Regression")
plt.legend()
plt.show()
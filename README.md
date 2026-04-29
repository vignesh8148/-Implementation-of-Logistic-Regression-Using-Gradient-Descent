# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Data Preparation: Load the dataset, map categorical labels to numerical values (0 and 1), and normalize input features using StandardScaler to ensure stable convergence.
2.Initialization: Set the weights (theta) to zero and define hyperparameters, including the learning rate (\alpha) and the number of iterations for the gradient descent process.
3.Forward Propagation: Use the sigmoid function to map the linear combination of inputs and weights into a probability value between 0 and 1.
4.Optimization: Iteratively update the weights by calculating the gradient of the cost function (log-loss) and moving in the opposite direction to minimize the error.
5.Prediction & Evaluation: Apply a decision threshold of 0.5 to the final probabilities to classify outcomes and calculate the model's overall classification accuracy.
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vignesh.k
RegisterNumber:  212225240183
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

 
data = pd.read_csv("Untitled.csv")
 
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})
 
X = data[['ssc_p', 'mba_p']].values
y = data['status'].values

 
scaler = StandardScaler()
X = scaler.fit_transform(X)

m = len(y)
X = np.c_[np.ones(m), X]
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
def cost_function(X, y, theta):
    h = sigmoid(X @ theta)
    return (-1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h))
 
theta = np.zeros(X.shape[1])
alpha = 0.1
cost_history = []

for i in range(500):
    z = X @ theta
    h = sigmoid(z)
    gradient = (1/m) * X.T @ (h - y)
    theta = theta - alpha * gradient
    
    cost = cost_function(X, y, theta)
    cost_history.append(cost)

y_pred = (sigmoid(X @ theta) >= 0.5).astype(int)

accuracy = np.mean(y_pred == y) * 100
print("Weights:", theta)
print("Accuracy:", accuracy, "%")

plt.figure()
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Logistic Regression using Gradient Descent")
plt.show()
*/
```

## Output:
 <img width="805" height="624" alt="Screenshot 2026-04-29 110627" src="https://github.com/user-attachments/assets/dd9905ce-8648-4fdd-99b4-19a6f979913a" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.


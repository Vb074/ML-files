# This is a sample Python script.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('Salary_vs_Age.csv')

X = df[['Age']].to_numpy()
y = df['Salary '].to_numpy()

#Split
X_train = X[:10]
y_train = y[:10]

X_val = X[10:20]
y_val = y[10:20]

X_test = X[20:]
y_test = y[20:]

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

#Plotting a quadratic regression
poly2 = PolynomialFeatures(degree = 2)

X_train_poly2 = poly2.fit_transform(X_train)
X_val_poly2 = poly2.transform(X_val)

quad = LinearRegression()
quad.fit(X_train_poly2, y_train)

y_train_pred = quad.predict(X_train_poly2)
y_val_pred = quad.predict(X_val_poly2)

train_rmse = rmse(y_train, y_train_pred)
val_rmse = rmse(y_val, y_val_pred)

X_test_poly2 = poly2.transform(X_test)
y_test_pred = quad.predict(X_test_poly2)
test_rmse = rmse(y_test, y_test_pred)

#Plotting
X_grid = np.linspace(X.min(), X.max(), 300).reshape(-1,1)
X_grid_poly2 = poly2.transform(X_grid)
y_grid = quad.predict(X_grid_poly2)

plt.scatter(X_train, y_train, color = 'red', label = 'Training Data')
plt.scatter(X_val, y_val, color = 'blue', label = 'Validation Data')
plt.plot(X_grid, y_grid, color = 'green', label = 'Linear Regression')
plt.title('Linear Regression')
plt.show()






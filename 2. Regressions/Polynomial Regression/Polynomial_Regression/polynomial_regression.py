# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:47:45 2020

@author: Shihab Sikder
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X,y)

#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
regressor2 = PolynomialFeatures(degree = 3)
X_poly = regressor2.fit_transform(X)
lin_regressor2 = LinearRegression()
lin_regressor2.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,regressor1.predict(X),color='blue')
plt.title('Prediction By Linear Regression')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_regressor2.predict(regressor2.fit_transform(X_grid)),color='blue')
plt.title('Prediction By Polynomial Regression')
plt.xlabel('Position Label')
plt.ylabel('Salary')
plt.show()

 



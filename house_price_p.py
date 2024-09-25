import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

dataset = pd.read_csv(r"C:\Users\91976\OneDrive\Desktop\GitHub\House Price Prediction\House_data.csv")
space=dataset['sqft_living']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

pred = regressor.predict(xtest)

plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

sqf_600 = regressor.predict([[600]])
print(f"predicted house price: ${sqf_600[0]:,.2f}")

bias = regressor.score(xtrain,ytrain)
variance = regressor.score(xtest, ytest)
train_mse = mean_squared_error(ytrain, regressor.predict(xtrain))
test_mse = mean_squared_error(ytest, pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

filename = 'House_price.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as House price.pkl")

import os
print(os.getcwd())

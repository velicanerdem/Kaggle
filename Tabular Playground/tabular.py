"""
    Options:
        Linear Regression
        Quadratic Regression
        Gaussian Process
        Neural Network
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import torch

type = "neural_network"
scale_data = True
enhance_quadratic = type == "quadratic_regression" or type == "ridge_quadratic" or type == "gaussian_quadratic"

data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")
test_data_id = test_data["id"]
y = data["target"]
train_data = data.drop(["target"], axis=1)

if enhance_quadratic:
    transformer = PolynomialFeatures(degree=3)
    transformer.fit(train_data)
    train_data = transformer.transform(train_data)
    test_data = transformer.transform(test_data)

if scale_data:
    scaler = StandardScaler()
    scaler.fit(train_data)
    data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

if type == "linear_regression" or type == "quadratic_regression":
    clf = LinearRegression()
elif type == "ridge_linear" or type == "ridge_quadratic":
    clf = Ridge(alpha=1)
elif type == "gaussian_process" or type == "gaussian_quadratic":
    clf = GaussianProcessRegressor()
elif type == "neural_network":
    clf = MLPRegressor(hidden_layer_sizes=(30, 5))


"""
results = open("results.txt", "a+")

score = cross_val_score(clf, data, y)

extra = ""
if extra == "":
    line = type + " resulted in: " + str(score)
else:
    line = type + " with " + extra + " resulted in: " + str(score)
results.write(line + "\n")
results.close()
print(score)

"""
clf.fit(train_data, y)
predictions = clf.predict(test_data)

final = pd.DataFrame(index=test_data_id, columns=["target"], data=predictions)
final.to_csv("predictions/" + type + "_result.csv")

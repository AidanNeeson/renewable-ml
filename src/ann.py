"""Artifical Neural Network Trained on Solar Data"""

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("../data/solar.csv")

X = df.loc[:, [False, True, True, True, False, True, True, False, False]]
y = df.loc[:, [False, False, False, False, False, False, False, True, True]]

X_train = X[:8000]
X_test = X[8000:]
y_train = y[:8000]
y_test = y[8000:]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(3,), random_state=0, max_iter=10000000)
reg.fit(X_train,y_train)
test = reg.predict(X_test)
print(test)
print(y_test)
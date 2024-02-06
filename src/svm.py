"""Support Vector Machines Trained on Solar Data"""

import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("../data/solar.csv")

X = df.loc[:, df.columns[1:3]]
y = df.loc[:, df.columns[3]]

X_train = X[:8000]
X_test = X[8000:]
y_train = y[:8000]
y_test = y[8000:]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = svm.SVR()
reg.fit(X_train, y_train)
test = reg.predict(X_test)
print(test)
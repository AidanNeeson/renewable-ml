"""Artifical Neural Network Trained on Solar Data"""

import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("../data/path-to-data-file.csv")
df = df.sample(frac=1)


X = df.loc[:, ['lat','long','capacity']]
y = df.loc[:, ['generated_energy','cost']]

X_train = X[:100000]
X_test = X[100000:]
y_train = y[:100000]
y_test = y[100000:]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(4,), random_state=0, max_iter=10000000)
reg.fit(X_train,y_train)
preds = reg.predict(X_test)
print(preds)
print()

#| METRICS GATHERING
r2 = metrics.r2_score(y_test, preds,multioutput="raw_values")
rmse = metrics.root_mean_squared_error(y_test, preds,multioutput="raw_values")
mape = metrics.mean_absolute_percentage_error(y_test, preds,multioutput="raw_values")

print()
print("Metric\tScore")
print("-----------------------")
print(f"r2\t{r2}\nrmse\t{rmse}\nmape\t{mape}")
print("-----------------------")
print()

#| K-FOLD CROSS VALIDATION
kf = KFold(n_splits=10, random_state=0, shuffle=True)
kf_cv_score = cross_val_score(reg, X, y, cv=kf)
print("10-Fold Cross Validation Score")
print("-----------------")
print(kf_cv_score)
"""Support Vector Machines Trained on Solar Data"""

import pandas as pd
from sklearn import svm
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("../data/wind_refactored.csv")
df = df.sample(frac=1)

# X = df.loc[:, [False, False, False, False, False, False, True, False, False, True, False, False]]
# y = df['energy_generated(mwh)'].values
X = df.loc[:, [False, False, False, False, True, True, False, True, True, True, True, False]]
y = df['cost($)'].values


X_train = X[:100000]
X_test = X[100000:]
y_train = y[:100000]
y_test = y[100000:]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = svm.SVR()
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print(preds)
print()

#| METRICS GATHERING
r2 = metrics.r2_score(y_test, preds)
mse = metrics.mean_squared_error(y_test, preds)
rmse = metrics.root_mean_squared_error(y_test, preds)
mae = metrics.mean_absolute_error(y_test, preds)

print()
print("Metric\tScore")
print("-----------------------")
print(f"r2\t{r2}\nmse\t{mse}\nrmse\t{rmse}\nmae\t{mae}")
print("-----------------------")
print()

#| K-FOLD CROSS VALIDATION
# kf = KFold(n_splits=10, random_state=0, shuffle=True)
# kf_cv_score = cross_val_score(reg, X, y, cv=kf)
# print("10-Fold Cross Validation Score")
# print("-----------------")
# print(kf_cv_score)
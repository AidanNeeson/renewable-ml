"""Random Forest Regression Alanaysis on Solar Data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.read_csv("../data/wind_refactored.csv")
df = df.sample(frac=1)

#| OLD DATA
# X = df.loc[:, [False, False, False, False, False, True, False, False, False]]
# y = df.loc[:, [False, False, False, False, False,False, False, True, True]]


#| REFACTORED DATA
# X = df.loc[:, [False, False, False, False, True, False, True, True, True, False, False, False]]
# y = df['energy_generated(mwh)'].values
# X = df.loc[:, [False, False, False, False, True, True, False, True, True, True, True, False]]
# y = df['cost($)'].values

X = df.loc[:, [False, False, False, False, True, True, True, True, True, True, False, False]]
y = df.loc[:, [False, False, False, False, False, False, False, False, False, False, True, True]]

X_train = X[:100000]
X_test = X[100000:]
y_train = y[:100000]
y_test = y[100000:]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = RandomForestRegressor(random_state=0)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print(preds)
print()


#| FEATURE IMPORTANCE GRAPH
# Energy features
features = ['wind_speed(m/s)', 'capacity(mw)', 'capacity_factor', 'available_wind_power(mw)']

# Cost features
features = ['wind_speed(m/s)','lcoe($/mwh)','capacity','capacity_factor','available_wind_power(mw)','available_energy(mwh)']

importances = reg.feature_importances_
indices = np.argsort(importances)

print("Importances")
print('-----------')
print(importances)
print('-----------')

plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

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

# train_results = []
# test_results = []
# list_nb_trees = [5, 10, 15, 30, 45, 60, 80, 100]

# for nb_trees in list_nb_trees:
#     rf = RandomForestRegressor(n_estimators=nb_trees)
#     rf.fit(X_train, y_train)

#     train_results.append(metrics.mean_squared_error(y_train, rf.predict(X_train)))
#     test_results.append(metrics.mean_squared_error(y_test, rf.predict(X_test)))

# line1, = plt.plot(list_nb_trees, train_results, color="r", label="Training Score")
# line2, = plt.plot(list_nb_trees, test_results, color="g", label="Testing Score")

# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('MSE')
# plt.xlabel('n_estimators')
# plt.show()
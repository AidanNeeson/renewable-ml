"""Random Forest Regression Alanaysis on Geospatial Data"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import sklearn.metrics as metrics
from sklearn.model_selection import cross_validate, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import tree

scaler = StandardScaler()

df = pd.read_csv("../data/path-to-datafile.csv")
df = df.sample(frac=1)

X = df.loc[:, ['lat','long','capacity']]
y = df.loc[:, ['generated_energy','cost']]

# Wind
X_train = X[:100000]
X_test = X[100000:]
y_train = y[:100000]
y_test = y[100000:]

# Solar
# X_train = X[:9500]
# X_test = X[9500:]
# y_train = y[:9500]
# y_test = y[9500:]

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = RandomForestRegressor(random_state=0)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print(preds)
print(y_test)
print()

display = y_test.reset_index()

for i in range(3):
    print(f"predicted energy: {preds[i][0]:2f}\tactual energy: {display.at[i, 'generated_energy']:2f}\tpredicted cost: {preds[i][1]:2f}\tactual cost: {display.at[i, 'cost']:2f}")



#| FEATURE IMPORTANCE GRAPH
# Wind
# features = ['lat','long','capacity']

# Solar
# features = ['irradiance','lcoe','capacity_factor','array_area','available_solar_resource']

importances = reg.feature_importances_
indices = np.argsort(importances)

print("Importances")
print('-----------')
for i in indices:
    print(f"{features[i]}: {importances[i]*100}")
print('-----------')

plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()

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
kf_cv_scores = cross_validate(reg, X, y, cv=kf, scoring={"r2":metrics.make_scorer(score_func=metrics.r2_score),
 "rmse":metrics.make_scorer(score_func=metrics.root_mean_squared_error),
 "mape":metrics.make_scorer(score_func=metrics.mean_absolute_percentage_error)})
kf_cv_df = pd.DataFrame.from_dict(kf_cv_scores)
print("10-Fold Cross Validation Scores")
print("----------------------------------------------------")
print(kf_cv_df)


train_results = []
test_results = []
list_nb_trees = [5, 10, 15, 30, 45, 60, 80, 100]

for nb_trees in list_nb_trees:
    rf = RandomForestRegressor(n_estimators=nb_trees)
    rf.fit(X_train, y_train)

    train_results.append(metrics.mean_squared_error(y_train, rf.predict(X_train)))
    test_results.append(metrics.mean_squared_error(y_test, rf.predict(X_test)))

line1, = plt.plot(list_nb_trees, train_results, color="r", label="Training Score")
line2, = plt.plot(list_nb_trees, test_results, color="g", label="Testing Score")

plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('MSE')
plt.xlabel('n_estimators')
plt.show()

# Wind
# fn = ['wind_speed','lcoe','capacity','capacity_factor','available_wind_power','available_energy']

# # Solar
# # fn = ['irradiance','lcoe','capacity_factor','array_area','available_solar_resource'] 

cn = ['generated_energy','cost']
plt.subplots(nrows=1, ncols=1, figsize=(4,4), dpi=800)
tree.plot_tree(reg.estimators_[0],feature_names=fn,class_names=cn,filled=True, max_depth=3)
plt.show()

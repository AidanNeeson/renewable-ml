import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the wind dataset
wind_speeds = pd.read_csv('../data/wind_speeds.csv')

# Format locational wind speed data and split into training and testing
wind_speeds_X = list(zip(wind_speeds.lat, wind_speeds.long))
wind_speeds_X_train = wind_speeds_X[:500]
wind_speeds_X_test = wind_speeds_X[500:550]

# Split the targets into training/testing sets
wind_speeds_Y = wind_speeds['wind_speed']
wind_speeds_Y_train = wind_speeds_Y[:500]
wind_speeds_Y_test = wind_speeds_Y[500:550]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(wind_speeds_X_train, wind_speeds_Y_train)

# Make predictions using the testing set
wind_speeds_Y_pred = regr.predict(wind_speeds_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(wind_speeds_Y_test, wind_speeds_Y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(wind_speeds_Y_test, wind_speeds_Y_pred))

# Plot outputs
plt.scatter([str(i) for i in wind_speeds_X_test], wind_speeds_Y_test, color="black")
plt.plot(wind_speeds_X_test, wind_speeds_Y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
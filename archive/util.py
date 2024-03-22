"""Utility module that contains scripts used for dataset manipulaton throughout the project's lifetime."""
import pandas as pd
import csv
from geopy.geocoders import Nominatim

# Read in the dataset
df = pd.read_csv("../data/path-to-data-file.csv")

# Iteratively remove all instances of offshore wind turbines
for index,row in df.iterrows():
    if row.iloc[4] == 'offshore':
        df = df.drop([index])
df.to_csv('../data/path-to-data-file.csv', index=None, header=True)

# Map the LCOE for a given state to each data entry in the dataset
with open("../data/state_costs.csv") as file:
    costs = {}
    csv_reader = csv.reader(file)
    i = 0
    for row in csv_reader:
        if i != 0:
            costs[row[0]] = row[2]
        i += 1

i = 0
for index, row in df.iterrows():
    df.at[index, 'lcoe'] = costs[row['state']]
    print(i)
    i += 1

df.to_csv('../data/path-to-data-file.csv', index=None, header=True)

# Iteratively request the state based on location through geopy API call and add values to dataset
df['state'] = df['state'].astype(str)

cords = df.loc[:, df.columns[1:3]]

states = []
geolocator = Nominatim(user_agent="the_cool_place")

i = 0
for index, row in cords.iterrows():
    if i >= 0:
        coordinates = (row.iloc[0], row.iloc[1])
        location = geolocator.reverse(coordinates).raw
        if location['address']['country_code'] == "us":
            df.at[index, 'state'] = location['address']['state']
        print(i)
        df.to_csv('../data/path-to-data-file.csv', index=None, header=True)
    i += 1


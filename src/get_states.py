import pandas as pd
import csv
from geopy.geocoders import Nominatim

df = pd.read_csv("../data/wind_refactored.csv")

# for index,row in df.iterrows():
#     if row.iloc[4] == 'offshore':
#         df = df.drop([index])
# df.to_csv('../data/wind_refactored.csv', index=None, header=True)

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

df.to_csv('../data/wind_refactored.csv', index=None, header=True)


# df['state'] = df['state'].astype(str)

# cords = df.loc[:, df.columns[1:3]]

# states = []
# geolocator = Nominatim(user_agent="the_cool_place")

# i = 0
# for index, row in cords.iterrows():
#     if i >= 83356:
#         coordinates = (row.iloc[0], row.iloc[1])
#         location = geolocator.reverse(coordinates).raw
#         if location['address']['country_code'] == "us":
#             df.at[index, 'state'] = location['address']['state']
#         print(i)
#         df.to_csv('../data/wind_refactored.csv', index=None, header=True)
#     i += 1


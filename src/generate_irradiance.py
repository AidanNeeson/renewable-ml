import csv
from pypvwatts import PVWatts

PVWatts.api_key = 'i2h8CT4wpIBrtR7O2OdtNMo6cgBB6v1NCuBBe4d0'

csvwrite = open('../data/irradiance.csv', 'a', newline='')
writer = csv.writer(csvwrite)

with open('../data/wind_speeds.csv') as csvfile:
    reader = csv.reader(csvfile)
    i = 0
    for row in reader:
        if i == 0 or row[3] == 'offshore':
            i += 1
        else:
            i += 1
            if i >= 15025:
                try:
                    result = PVWatts.request(
                        system_capacity=4, module_type=1, array_type=1,
                        azimuth=190, tilt=30, dataset='nsrdb',
                        losses=13, lat=float(row[0]), lon=float(row[1]))
                    row[2] = result.solrad_annual
                    writer.writerow(row[0:4])
                except KeyError:
                    print('skipped broken location...')

        

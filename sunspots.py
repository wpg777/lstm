import numpy as np
import pandas as pd

WINDOW=10
TEST_DATA=40

df = pd.read_csv('../SN_y_tot_V2.0.csv', header = 0, delimiter = ";")
print(df)
x_train = np.random.random((10, 5))
years = df['year']
sunspots = df['average']

print(df['year'])
print(x_train)
print(years[0])
x_years = []
x_sunspots = []
y_sunspots = []
for i in xrange(WINDOW, len(years) - WINDOW - TEST_DATA):
  print(i)
  x_years.append(years[i-WINDOW:i])
  x_sunspots.append(sunspots[i-WINDOW:i])


import numpy as np
import pandas as pd

WINDOW=10

df = pd.read_csv('../SN_y_tot_V2.0.csv', header = 0, delimiter = ";")
print(df)
x_train = np.random.random((10, 5))
years = df['year']

print(df['year'])
print(x_train)
print(years[0])

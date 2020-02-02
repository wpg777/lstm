from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WINDOW=40
TEST_DATA=40
LSTM_DIM=700
DENSE_DIM=70
EPOCHS=800

df = pd.read_csv('../SN_y_tot_V2.0.csv', header = 0, delimiter = ";")
x_train = np.random.random((10, 5))
years = df['year']
sunspots = df['average']

def train():
  train_years = []
  train_sunspots = []
  train_x = []
  train_y = []
  y_sunspots = []
  for i in xrange(WINDOW, len(years) - WINDOW - TEST_DATA):
#  print(i)
    train_years.append(years[i-WINDOW:i])
    train_sunspots.append(sunspots[i-WINDOW:i])
    train_x.append(sunspots[i-WINDOW:i-1])
    train_y.append(sunspots[i])

  train_years = np.array(train_years)
  train_years = np.reshape(train_years, (train_years.shape[0], 1, train_years.shape[1]))
  train_sunspots = np.array(train_sunspots)
  train_sunspots = np.reshape(train_sunspots, (train_sunspots.shape[0], 1, train_sunspots.shape[1]))
  train_x = np.array(train_x)
  train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
  train_y = np.array(train_y)
  train_y = np.reshape(train_y, (train_y.shape[0], 1))
  print(train_x[0])
  print(train_y[0])
  # define and fit the final model
  model = models.Sequential()
  #model.add(Input(shape=train_sunspots.shape()))
  #model.add(Dense(4, input_dim=WINDOW, activation='relu'))
#  model.add(LSTM(128, input_shape=(1, WINDOW-1), return_sequences=True))
#  model.add(Dense(20, activation='relu'))
  model.add(layers.LSTM(LSTM_DIM, input_shape=(1, WINDOW-1), return_sequences=True))
  model.add(layers.LSTM(LSTM_DIM, input_shape=(1, WINDOW-1), return_sequences=True))
  model.add(layers.LSTM(LSTM_DIM, input_shape=(1, WINDOW-1), return_sequences=True))
  model.add(layers.LSTM(LSTM_DIM, input_shape=(1, WINDOW-1)))
  model.add(layers.Dense(DENSE_DIM, activation='relu'))
#  model.add(Dropout(0.15))
#  model.add(Flatten())
  model.add(layers.Dense(1, activation='linear'))
  model.compile(loss='mse', optimizer='adagrad')
  model.fit(train_x, train_y, epochs=EPOCHS, verbose=2)
  model.save('mse-adam-{}-{}-{}.h5'.format(LSTM_DIM, WINDOW, DENSE_DIM))
#print(len(train_years))
#print(len(train_sunspots))

#print(train_years)
#print(train_sunspots)
#print(train_years)
#print(train_sunspots)


def eval():
  test_years = []
#  test_x = []
  test_y = []
#  for i in xrange(len(years) - WINDOW - TEST_DATA + 1, len(years)):
  for i in xrange(len(years)):
    test_years.append(years[i])
#    test_x.append(sunspots[i-WINDOW:i-1])
    test_y.append(sunspots[i])
#  test_x = np.array(test_x)
  sunspot_values = test_y
  test_y = np.array(test_y)
#  test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
#  print(test_x.shape)
  model = load_model('mse-adam-512-40-70.h5')
#  pred_y = model.predict(test_x, verbose=1)
#  print(test_y)
#  print(pred_y)
#  print(test_years)
#  matplotlib.use('Agg')
#  plt.plot(test_years, test_y)
#  plt.plot(test_years, pred_y)
#  plt.legend(['Actual sunspots.', 'Sunspot predictions'])
#  plt.show()

  pred_sunspots = np.array(sunspots[-WINDOW+1:]).tolist()
  print(pred_sunspots)
  for i in xrange(100):
    test_years.append(test_years[-1]+1)
    window = np.array(pred_sunspots[-WINDOW+1:])
#    print(i)
#    print(window)
    window_input = np.reshape(window, (1, 1, window.shape[0]))
#    print(window_input)
    pred = model.predict(window_input, verbose=1)
    print(pred)
    pred_sunspots.append(pred[0][0])
    sunspot_values.append(pred[0][0])
#    test_y.append(pred[0][0])
  full_pred = test_y.tolist() + pred_sunspots
#  print(full_pred)
  print(len(sunspot_values))
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)

  # Major ticks every 20, minor ticks every 5
  x_major_ticks = np.arange(1700, 1700+len(sunspot_values), 50)
  x_minor_ticks = np.arange(1700, 1700+len(sunspot_values), 10)
  y_major_ticks = np.arange(0, 300, 50)
  y_minor_ticks = np.arange(0, 300, 10)

  ax.set_xticks(x_major_ticks)
  ax.set_xticks(x_minor_ticks, minor=True)
  ax.set_yticks(y_major_ticks)
  ax.set_yticks(y_minor_ticks, minor=True)

  # And a corresponding grid
  ax.grid(which='both')

  # Or if you want different settings for the grids:
  ax.grid(which='minor', alpha=0.2)
  ax.grid(which='major', alpha=0.5)
  plt.plot(xrange(1700, 1700+len(sunspot_values)), sunspot_values)
  plt.axvline(x=2017, color='k', linestyle='--')
  plt.show()
#eval()
train()

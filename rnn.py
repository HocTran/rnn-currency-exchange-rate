# python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:01:13 2021

@author: hoctran
"""

# Part 1 - Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Constants

# number of timesteps
TIME_STEPS = 120
# Number of neurons in LSTM
LSTM_UNITS = 100
# Rate that LSTM will drop when backpropagating
DROPOUT = 0.1
# Number of times the model training
EPOCHS = 100
# Number of data to feed in a batch
BATCH_SIZE = 32

#1. Importing the training set
dataset_train = pd.read_csv('rate_train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#2. Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

#3. Creating a data structure with TIME_STEPS timesteps and 1 output
X_train = []
y_train = []
for i in range(TIME_STEPS, len(training_set)):
    X_train.append(training_set_scaled[i-TIME_STEPS:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


#4 Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 - Building the RNN

#1. Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#2. Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(DROPOUT))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True))
regressor.add(Dropout(DROPOUT))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = LSTM_UNITS, return_sequences = True))
regressor.add(Dropout(DROPOUT))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = LSTM_UNITS))
regressor.add(Dropout(DROPOUT))

# Adding the output layer
regressor.add(Dense(units = 1))

#3. Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#4. Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)



# Part 3 - Making the predictions and visualising the results

#1. Getting the real exchange rate (in test set)
dataset_test = pd.read_csv('rate_test.csv')
real_exchange_rates = dataset_test.iloc[:, 1:2].values
dateStrings = dataset_test['date'].values

#2. Predicte exchange rate
dataset_total = pd.concat((dataset_train['rate'], dataset_test['rate']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - TIME_STEPS:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(TIME_STEPS, TIME_STEPS + len(dataset_test)):
    X_test.append(inputs[i-TIME_STEPS:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_rate = regressor.predict(X_test)
predicted_rate  = sc.inverse_transform(predicted_rate)

# Part 4 - Visualising the results
dates = list(map(lambda x: datetime.strptime(x, '%Y.%m.%d'), dateStrings))

fig, ax = plt.subplots(sharex=True, sharey=True)
fig.autofmt_xdate()

ax.plot(dates, list(real_exchange_rates), color = 'red', label = 'Real')
ax.plot(dates, list(predicted_rate), color = 'blue', label = 'Prediction')

plt.title('USD->THB Exchange Rate Prediction')
plt.legend()
plt.show()
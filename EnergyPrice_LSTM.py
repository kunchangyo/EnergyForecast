#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 12:36:30 2017

@author: kunchang
"""


# coding: utf-8

# In[14]:

import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# In[13]:

## Prediction of 5-day 
# Upload data
import math
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import Flatten

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# In[20]:

def normalize_data(df):
    """
    Scale Electricity Price to be within [0,1]: Pn = (P-Pmin)/(Pmax-Pmin)
    Original levels are not stationary
    Args: 
        Electricity price dataFrame contains price only without time labels.
    Returns:
        Normalized DataFrame.
    """
    df = df.values
    df = df.astype('float32')
    df_result = []
    for i in range(df.shape[1]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        temp_df = scaler.fit_transform(df[:,i])
        df_result.append(temp_df)
    df_result = pd.DataFrame(np.transpose(df_result))    
    return df_result


# In[21]:

def scale_back(df, df_scaled):
    """
        Scale predicted values back to original levels for comparison
    Args: 
        predictions (predicted price)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    temp = scaler.fit_transform(df)
    df_orig = scaler.inverse_transform(df_scaled)
    return df_orig


# In[22]:

def create_rolling_sequences(df, window_size, time_lag, labels=False):
    """
    creates new data frame based on previous observation
    """
    rnn_df = []
    for i in range(len(df) - window_size + 1):
        if labels:
            try:
                rnn_df.append(df.iloc[i + window_size].as_matrix())
            except AttributeError:
                rnn_df.append(df.iloc[i + window_size])
        else:
            data_ = df.iloc[i: i + window_size].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    df_rolling = np.array(rnn_df, dtype=np.float32)
    features_ = df_rolling[:-time_lag] # features enter into RNN needs to be 3D
    labels_ = df_rolling[time_lag:]
    labels_r = labels_[:,:,:1] # take first value of the feature set: price labels
    labels_r = np.squeeze(labels_r,axis=2) # reshape labels to 2D

    return np.array(features_), np.array(labels_r)


# In[23]:

###### Test sequence func
df2 = full_df[:24]
features_, labels_ = create_rolling_sequences(df2, window_size=1, time_lag=1, labels=False)
print df2
print features_.shape
print labels_.shape
print features_[-2:]
print labels_[-2:]


# In[24]:

def create_training_and_test(df, window_size, time_lag, num_days_test):
    """
      Split data into training and testing
    Args: 
        df: DataFrame.
        forecast_horizon (int): how many days to predict; used to create proper sequences. 
        time_lag (int): if time_lag = 1, using X_t to predict X_t+1. "t" is sequence-wise.
        num_days_test: how many days to leave for testing. If num_days_test=3 and forecast_horizon=5, it leaves 15 days for testing
    Returns:
        Numpy arrays.
    """
    num_seq_test = num_days_test*24 # translate days into hours with consideration of window_size
    features_, labels_ = create_rolling_sequences(df, window_size, time_lag)
    features_train = features_[:-num_seq_test]
    features_test = features_[-num_seq_test:]
    labels_train = labels_[:-num_seq_test]
    labels_test = labels_[-num_seq_test:]
    return features_train, labels_train, features_test, labels_test


# In[25]:

# Test split function
features_train, labels_train, features_test, labels_test = create_training_and_test(full_df, window_size=120, time_lag=1, num_days_test=14)
print features_train.shape
print labels_train.shape


# In[186]:

def LSTM_Model(df, window_size, time_lag, num_days_test, num_hidden_units, learning_rate, batch_size):
    """
      The current setting has one hidden layer controlled by LSTM, and one fully-connected layer controlled by Dense.
      Activation, loss and optimizer can be changed.
    """
    features_train, labels_train, _, _ = create_training_and_test(df, window_size, time_lag, num_days_test)
    # Construct layers
    LSTM_model = Sequential() # Initialize: this is the 1st layer in the sequential model
    LSTM_model.add(LSTM(num_hidden_units[0], input_shape=(window_size, features_train.shape[2]), return_sequences=True))
    #LSTM_model.add(LSTM(num_hidden_units[0], batch_input_shape=(1, window_size, features_train.shape[2]),stateful=True))
    #LSTM_model.add(LSTM(1, input_shape=(window_size, time_lag)))
    #LSTM_model.add(RepeatVector(window_size)) # Apply 2nd layer for many-to-many 
    LSTM_model.add(LSTM(num_hidden_units[0], return_sequences=True))
    LSTM_model.add(LSTM(num_hidden_units[0]))
    LSTM_model.add(Dense(window_size)) # Dense: fully-connected. "TimeDistributed" 
    LSTM_model.add(Activation('linear'))
    SGD = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    #LSTM_model.compile(loss='mean_squared_error', optimizer=SGD)
    LSTM_model.compile(loss='mean_absolute_error', optimizer=SGD)
    LSTM_model.fit(features_train, labels_train, epochs=20, batch_size=batch_size, verbose=2)

    return LSTM_model


# In[107]:

# test rolling forecast
df2 = full_df_scale[:2*24]
#print full_df.iloc[23]
cutoff = 1*24
temp_df = df2[:-cutoff]
temp_df.astype('float32')
labels_rolling_pred = []
print np.array(temp_df).shape


# In[170]:

def rolling_forecast(df, window_size, time_lag, num_days_test, LSTM_train):
  cutoff = num_days_test*24
  temp_df = df[:-cutoff]
  temp_df.astype('float32')
  labels_rolling_pred = []
  
  for ii in range(cutoff):
    features_train_temp, _ = create_rolling_sequences(temp_df, window_size, time_lag, labels=False)
    next_pred = LSTM_train.predict(features_train_temp[-1:], batch_size=1) 
    next_pred = next_pred[-1,-1] # only last number is our forecast. work for window_size>1, if window_size=1, need np.squeeze(next_pred, axis=0)    
    labels_rolling_pred.append(next_pred)

    # Replace last row's true price with price_hat
    hat_feature_row_s = df.iloc[(-cutoff+ii-1):(-cutoff+ii), -6:]
    hat_feature_row_s = hat_feature_row_s.values
    hat_feature_row = np.insert(hat_feature_row_s, [0], next_pred, axis=1)

    # Add one new row in order to create rolling sequence(not used for forecast though, the new row will go to new label)
    temp_df2 = temp_df[1:-1].values
    temp_df3 = np.append(temp_df2, hat_feature_row, axis=0)
    if ii == cutoff-1:
      next_feature_row = df.iloc[[-1]].values
    else:
      next_feature_row = df.iloc[(-cutoff+ii):(-cutoff+ii+1)].values
    temp_df4 = np.append(temp_df3, next_feature_row, axis=0)
    temp_df = pd.DataFrame(temp_df4)

    #output = np.array([ii, next_pred])
    #print output
  return labels_rolling_pred


# In[234]:

def MAE_Nd_best(pred_window, labels_pred, labels_test):
  num_days_pred = len(labels_test)/24
  MAE_list = []
  for ii in range(num_days_pred-pred_window+1):
    MAE_d = mean_absolute_error(labels_test[(ii*24):(ii+pred_window)*24], labels_pred[(ii*24):(ii+pred_window)*24])
    MAE_list.append(MAE_d)
  MAE_sort = np.sort(MAE_list)
  result = np.mean(MAE_sort[:-(pred_window-1)])
  return result


# In[235]:

# Prepare 'labels_test' and 'labels_pred': Inverse the predictions from [0, 1] to original levels
def get_forecast(df_work, df_orig, window_size, time_lag, num_days_test, LSTM_train, pred_window):
  labels_pred_scale = rolling_forecast(df_work, window_size, time_lag, num_days_test, LSTM_train)
  labels_pred = scale_back(df_orig, labels_pred_scale)
  labels_test = df_orig[-num_days_test*24:].values
  # Calculate MAE
  MAE_test = mean_absolute_error(labels_test, labels_pred)
  MAE_5d_best = MAE_Nd_best(pred_window, labels_pred, labels_test)
  
  print('Overall MAE on testing: %.2f' % (MAE_test))
  print('Best MAE on 5d rolling: %.2f' % (MAE_5d_best))
  
  return labels_pred, labels_test


# In[ ]:

# Load data
energy_price = pd.read_csv("ElectricityPrice.csv", sep=";")
historical_weather = pd.read_csv("historical_weather.csv", sep=",")
points = pd.read_csv("points.csv", sep=",")
print historical_weather.head(4)

# Extract electricity market price without time labels
price_df = energy_price['Price']
#print price_df.head(5)

# points are evenly distributed
points.head(8)
#plt.scatter(np.array(points["longitude"]), np.array(points["latitude"]))

# Approach 1: Averaging features accross 18 points
historical_weather_avg = pd.DataFrame()
historical_weather_avg = historical_weather.groupby('prediction_date', group_keys=False).mean().reset_index() #this also takes average of points which is meaningless to us but faster
#print historical_weather_avg.head(4)
#print historical_weather_avg.shape

# Then merge with electrcity price date by prediction_date
energy_and_weather = pd.merge(energy_price, historical_weather_avg, left_on = 'date (UTC)', right_on='prediction_date')
print energy_and_weather.shape
energy_and_weather.head(2)

# Create data frame of both weather and prices
# Extract electricity market price without time labels
price_df = energy_and_weather['Price']
weather_df = energy_and_weather[['temperature', 'air_density', 'pressure', 'wind_gust', 'wind_direction']]
full_df = energy_and_weather[['Price', 'temperature', 'air_density', 'pressure', 'precipitation','wind_gust', 'wind_direction']]
full_df_scale = normalize_data(full_df)
print full_df_scale.shape


# In[243]:

labels_pred, labels_test = get_forecast(full_df_scale, price_df, 30, 1, 18, LSTM_train,5)


# In[242]:

# Train LSTM
LSTM_train = LSTM_Model(
  df = full_df_scale,
  window_size = 30, # measure in num of hours
  time_lag = 1, 
  num_days_test = 18, # leave 14 days for out-of-sample prediction/testing
  num_hidden_units = np.array([100]), # 2 layers 
  learning_rate = 0.1,
  batch_size = 100
)


# In[244]:

# Plot real vs. predicted
sb.set_style("whitegrid")
plt.plot(labels_test, sb.xkcd_rgb["denim blue"], lw=2, label='Real Price')
plt.plot(labels_pred, sb.xkcd_rgb["medium green"], lw=1.5, label='Predicted Price (LSTM ts=30h)')
plt.xlabel('Time (hours)'); plt.ylabel('Electricity Price')
plt.title('Real vs. Predicted Hourly Electricity Price (MAE = 5.99)')
plt.legend(shadow=True, fancybox=True)
plt.ylim([20, 80])
plt.show()


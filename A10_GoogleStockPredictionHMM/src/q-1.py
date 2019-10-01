#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt


# In[13]:


def fun(layers, cells, time_stamp, inp, op):
    print("For RNN containing "+str(layers)+" hidden layers "+str(cells)+" cells and "+str(time_stamp)+" time stamp.")
    inp_scale = MinMaxScaler(feature_range=(0, 1))
    scaled_inp = inp_scale.fit_transform(inp)
    
    out_scale = MinMaxScaler(feature_range=(0,1))
    scaled_op = out_scale.fit_transform(op)
    
    x = []
    y = []
    for i in range(time_stamp, len(inp)):
        x.append(scaled_inp[i-time_stamp:i,:])
        y.append(scaled_op[i,0])
    
    x = np.array(x)
    y = np.array(y)
    y.shape = (len(y), 1)
    
    print(x.shape)
    print(y.shape)
    ind = int(0.8 * len(x))
    print(ind)
#     x_tr, x_tes, y_tr, y_tes = train_test_split(x, y, test_size = 0.2, random_state = 20)
    x_tr = x[:ind]
    x_tes = x[ind:]
    y_tr = y[:ind]
    y_tes = y [ind:]
    
    print("X_tr : ",x_tr.shape)
    print("Y_tr : ",y_tr.shape)
    print("X_tes : ",x_tes.shape)
    print("Y_tes : ",y_tes.shape)

    rnn_model = Sequential()
    rnn_model.add(LSTM(units= cells, return_sequences= True, input_shape = (x_tr.shape[1], x_tr.shape[2])))
    rnn_model.add(Dropout(0.2))
    
    rnn_model.add(LSTM(units= cells))
    rnn_model.add(Dropout(0.2))
    
    rnn_model.add(Dense(units= 1))
    
    rnn_model.compile(optimizer = Adam(0.01), loss = "mean_squared_error")
    rnn_model.fit(x_tr, y_tr, epochs = 100, batch_size = 50, verbose = 0)
    
    predicted = rnn_model.predict(x_tes)
    predicted = out_scale.inverse_transform(predicted)
#     real = out_scale.inverse_transform(y_tes)
    real = op[ind + time_stamp:]
    
    plt.plot(real, color = 'red', label = 'real_price')
    plt.plot(predicted, color = 'blue', label = 'predicted_price')
    plt.title('Stock prediction with Hidden Lyaers = %d, No. of cells = %d, and timestamp = %d'%(layers, cells, time_stamp))
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    


# In[14]:


def avg_low_high():
    no_of_hidden = [2,3]
    no_of_cells = [20, 30, 50]
    time_stamps = [20, 50 ,75]
    
    frame = pd.read_csv("../input/GoogleStocks.csv")
    frame = frame.drop(frame.index[0])
    frame = frame.convert_objects(convert_numeric=True)
    
    df = frame[['volume']]
    
    high = frame.iloc[:,4:5].values
    low = frame.iloc[:,5:6].values
    avg = (low + high)/2
    df['avg'] = avg
    opening = frame.iloc[:,3:4].values
    
    for h in no_of_hidden:
        for c in no_of_cells:
            for t in time_stamps:
                fun(h, c, t, avg, opening)
#     fun(2, 20, 20, df, opening)

avg_low_high()


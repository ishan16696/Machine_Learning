#!/usr/bin/env python
# coding: utf-8

# ### Question 1-2

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM


# In[ ]:


def create_markov(layers):
    markov = GaussianHMM(n_components= layers, covariance_type="diag", n_iter=500)
    markov.fit(X)
    calculated = calculate(markov)
    return markov, calculated


# In[ ]:


def calculate(markov):
    exp = np.dot(markov.transmat_, markov.means_)
    ret = list(zip(*exp))
    return ret


# In[ ]:


def predict_prices(timestamp, hidden, calculated):
    print(calculated)
    ls = []
    n = timestamp
    price = 0.0
    flag = False
    for i in range(length):
        if(i>n):
            pass
        else:
            st = hidden[-n+i]
            cur = opening[-n+i]
            print(st)
            ls.append(cur - calculated[st])


# In[ ]:


frame = pd.read_csv("GoogleStocks.csv")
frame = frame.drop(frame.index[0])
frame = frame.convert_objects(convert_numeric=True)

length = len(frame)

df = frame[['volume']]

high = frame.iloc[:,4:5].values
low = frame.iloc[:,5:6].values
avg = (low + high)/2

df['avg'] = avg

opening = frame['open'].values

X = df.values

no_of_hidden = [4, 8, 12]
no_of_timesteps = [20 ,50 ,75]

for h in no_of_hidden:
    for t in no_of_timesteps:
        model, calculated = create_markov(h)
        hidden = model.predict(X)
        print(len(calculated))
        predicted = predict_prices(t, hidden, calculated)
        plt.plot(predicted, color= 'red', label = 'predicted')
        plt.plot(opening, color= 'blue', label = 'Real')
        plt.legend()
        plt.show()


# ### Question 2

# **<font color = "red">- This is the solution of the question number 2</font>**

# <img src = "img1.jpeg">

# <img src = "img2.jpeg">

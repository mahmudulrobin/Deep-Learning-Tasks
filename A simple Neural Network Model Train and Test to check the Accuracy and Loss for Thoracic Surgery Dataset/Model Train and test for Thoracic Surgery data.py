# -*- coding: utf-8 -*-
"""lab 4 task(DNN)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U4D4pwU24GoS0wVC-Z7VNTsoEtHquVSn
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data= np.loadtxt("thoracic_surgery.csv", delimiter=",")
X=data[:, 0:17]
y=data[:, 17]

model = Sequential()
model.add(Dense(30, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, epochs=100, batch_size=10)
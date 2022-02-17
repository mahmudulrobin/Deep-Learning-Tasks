# -*- coding: utf-8 -*-
"""lab10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16rWcCgCo0hOxF0sNZEI25OI7Ws2hZfzL
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.array([[2.3,6.13],[1.2,4.71],[4.3,11.13],[5.7,14.29],[3.5,9.54],[8.9,22.43]])

x=data[:,0]
y=data[:,1]

w,b=0,0
alpha=0.05

for i in range(2000):
  w= w - alpha * (1/len(data)) * sum((w*x+b -y)*x)
  b= b - alpha * (1/len(data)) * sum(w*x+b -y)

print(w,"  ",b)
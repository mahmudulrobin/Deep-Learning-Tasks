# -*- coding: utf-8 -*-
"""lab8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zLF8DNhEqhMx2Kr0bbCajL9lHjefwj4U
"""

import numpy as np
a_b= np.array([3,4])
data=np.array([[2.3,6.13],[1.2,4.71],[4.3,11.13],[5.7,14.29],[3.5,9.54],[8.9,22.43]])
x=data[:,0]
y=data[:,1]
MSE=sum(((a_b[0]*x + a_b[1])-y)**2)/6

print(MSE)

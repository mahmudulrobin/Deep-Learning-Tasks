from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.

from keras import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.


(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float32')/255
testX = testX.reshape(testX.shape[0], 28, 28, 1).astype('float32')/255

#momentum
# keras.optimizer.SGD(lr=0.1, momentum=0.9)

#Nesterov Momentum
# keras.optimizer.SGD(lr=0.1, momentum=0.9, nesterov=True)

#adagrad
#keras.optimizer.Adagrad(lr=0.01, epsilon=1e-6)

#RMSprop
#keras.optimizer.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08,decay=0.0)


#Adam - combine momentum and rmsprop
# keras.optimizer.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,epsilon=1e-08, decay=0.0)

trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)

# model = Sequential()
model = keras.Sequential()

# model.add(Conv2D(128, 3, padding='same', activation = 'sigmoid'))
# model.add(Dropout(0.1))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Conv2D(128, 3, padding='same', activation = 'sigmoid'))
# model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, padding='same', activation = 'sigmoid'))
# model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(32, 3, padding='same', activation = 'sigmoid'))
# model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))


# opt=tf.keras.optimizer.SGD(lr=0.1, momentum=0.9)
callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
#opt = keras.optimizers.Adam(learning_rate = 0.0001)
# model.compile(loss = 'categorical_crossentropy', optimizer=opt)
#
# model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #works
# sgd = tf.keras.optimizers.SGD(lr=0.01, clipnorm=1.)
# sgd = tf.keras.optimizers.SGD(lr=0.01, clipvalue=0.5)
# momentum=tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

# adagrad=tf.keras.optimizers.Adagrad(lr=0.01, epsilon=1e-6)

# RMSprop
rms=tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08,decay=0.0)
model.compile(optimizer=rms, loss = 'categorical_crossentropy', metrics = ['accuracy']) #works


# model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy']) # multiclass classification
history = model.fit(trainX, trainY, epochs = 100, batch_size = 800, callbacks=[callback], verbose = 2)

y_loss = history.history['accuracy']

plt.plot(np.arange(len(y_loss)), y_loss, marker = '.', c = 'blue')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('train accuracy')
plt.show()

print(model.evaluate(testX, testY)[1])
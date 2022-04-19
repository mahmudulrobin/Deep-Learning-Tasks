from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float32')/255
testX = testX.reshape(testX.shape[0], 28, 28, 1).astype('float32')/255


trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)


model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), input_shape=(28, 28, 1), activation = 'sigmoid'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Conv2D(128, kernel_size=(3, 3), activation = 'sigmoid'))
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))


callback = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)
#opt = keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy']) # multiclass classification
history = model.fit(trainX, trainY, epochs = 100, batch_size = 800, callbacks=[callback], verbose = 2)

y_loss = history.history['accuracy']

plt.plot(np.arange(len(y_loss)), y_loss, marker = '.', c = 'blue')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('train accuracy')
plt.show()

print(model.evaluate(testX, testY)[1])
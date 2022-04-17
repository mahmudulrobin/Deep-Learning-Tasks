from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np


(trainX, trainY), (testX, testY) = mnist.load_data()
# (trainX, trainY)=mnist_train.csv
# (testX, testY) = mnist.test.csv

trainX = trainX.reshape(trainX.shape[0], 28, 28, 1).astype('float32')/255
testX = testX.reshape(testX.shape[0], 28, 28, 1).astype('float32')/255


trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)


model = Sequential()
model.add(Conv2D(18, kernel_size=(3, 3), input_shape=(28, 28, 1), activation = 'sigmoid'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(18, kernel_size=(3, 3), activation = 'sigmoid'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))


model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy']) # multiclass classification
history = model.fit(trainX, trainY, epochs = 50, batch_size = 600)

y_loss = history.history['accuracy']

plt.plot(np.arange(len(y_loss)), y_loss, marker = '.', c = 'blue')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('train accuracy')
plt.show()

print(model.evaluate(testX, testY)[1])
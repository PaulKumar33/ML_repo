from keras.datasets import mnist
from keras import models
from keras import layers

(trainImages, trainLabel),(testImages,testLabels) = mnist.load_data()

network = models.Sequential()
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#reshaping the data
trainImages = trainImages.reshape((60000,28*28))
trainImages = trainImages.astype('float32')/255
testImages = testImages.reshape((10000,28*28))
testImages = testImages.astype('float32')/255

from keras.utils import to_categorical
trainLabel = to_categorical(trainLabel)
testLabels = to_categorical(testLabels)

network.fit(trainImages, trainLabel, epochs=5,batch_size=128)

testLoss, testAcc = network.evaluate(testImages, testLabels)
print("\n\n\n")
print("Test Accurracy: ", testAcc)
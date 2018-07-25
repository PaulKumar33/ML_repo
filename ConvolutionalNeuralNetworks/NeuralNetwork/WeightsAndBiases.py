from os import environ
from keras.models import load_model
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#LOADING MODEL
my_model= load_model(filepath=r'C:\Users\paulk\OneDrive - University of Toronto\engsci 1t9\python_scripts\ConvolutionalNeuralNetworks\model_save.h5')
print(my_model.summary(), '\n')

#showing the parameters of the model
print('last node biases: {0}'.format(my_model.get_weights()[-1]))
print('last node weigths: {0}'.format(my_model.get_weights()[-2]))

#showing the weights and biases of the
print('Amount of layers in the network: {0}'.format(my_model))
print('last node biases: {0}'.format(my_model.get_weights()[2]))
print('size of the array: {0}'.format(len(my_model.get_weights()[2][:])))


'''print('last node weigths: {0}'.format(my_model.get_weights()[-6]))
print('second node weights: {0}'.format(len(my_model.get_weights()[-6])))'''

#testing the new data on the network
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


#test_labels = to_categorical(test_labels)

#predicting a random number
test_images = test_images[0:1000]
test_labels = test_labels[0:1000]
test_images = test_images.reshape((1000, 28, 28, 1))

test_labels = to_categorical(test_labels)
(eval_l, eval_acc) = my_model.evaluate(test_images, y=test_labels, batch_size=5000)
print('Evaluation Accuracy is: {:4.2f}%'.format(eval_acc*100))

#creating a distribution of accuracy
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

accuracyArray = []
batchSize = []

for i in range(1000,10000,1000):
    print(i)
    test_images_hold = test_images[0:i]
    test_labels_hold = test_labels[0:i]
    test_images_hold = test_images_hold.reshape((i,28,28,1))
    test_labels_hold = to_categorical(test_labels_hold)
    (eval_l, eval_acc) = my_model.evaluate(test_images_hold, y=test_labels_hold, batch_size=i)

    batchSize.append(i)
    accuracyArray.append(eval_acc*100)
plt.plot(batchSize, accuracyArray)
plt.xlabel("Test Size")
plt.ylabel("Test Accuracy")
plt.xlim([0,11000])
plt.ylim([70,100])
plt.title("Accuracy vs test Size")
plt.show()



import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models, layers, optimizers
from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Loading datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('Train images shape:', train_images.shape)
print('Train labels shape:', train_labels.shape, '\n')
print('Test images shape:', test_images.shape)
print('Test labels shape:', test_labels.shape, '\n')

# Showing the First MNIST handwritten number
# plt.imshow(train_images[1], cmap='Greys')
# plt.show()
# print('Train image label shown:', train_labels[1], '\n')

# Training, Validation and Test datasets
valid_images = train_images[50000:60000]
valid_labels = train_labels[50000:60000]
train_images = train_images[0:50000]  # test images remain untouched
train_labels = train_labels[0:50000]

# Creating Tensors
train_images = train_images.reshape((50000, 28, 28, 1))
train_images = train_images.astype('float32') / 255  # converting to a [0, 1] scale
valid_images = valid_images.reshape((10000, 28, 28, 1))
valid_images = valid_images.astype('float32') / 255  # converting to a [0, 1] scale
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255  # converting to a [0, 1] scale

# One-hot encode labels
print('A label:', train_labels[19])
train_labels = to_categorical(train_labels)
valid_labels = to_categorical(valid_labels)
test_labels = to_categorical(test_labels)
print('A one-hot encode label:', train_labels[19])

# CNN Architecture
my_model = models.Sequential()
my_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
                           use_bias=True, input_shape=(28, 28, 1)))
my_model.add(layers.Activation('relu'))
my_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), use_bias=True))
my_model.add(layers.Activation('relu'))

my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

my_model.add(layers.Dropout(rate=0.2))

my_model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), use_bias=True))
my_model.add(layers.Activation('relu'))

my_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

my_model.add(layers.Flatten())

my_model.add(layers.Dropout(rate=0.2))

my_model.add(layers.Dense(units=10, use_bias=True))
my_model.add(layers.Activation('relu'))

my_model.add(layers.Dense(units=10, use_bias=True))
my_model.add(layers.Activation('softmax'))

# CNN Summary
my_model.summary()

# CNN Loss and Optimizer
compile = my_model.compile(optimizers.sgd(lr=0.1, decay=0.01), loss='categorical_crossentropy',
                           metrics=['accuracy'])

# CNN Training
fit = my_model.fit(x=train_images, y=train_labels, batch_size=2500, epochs=20,
                   validation_data=(valid_images, valid_labels))

# Save Model
my_model.save(filepath=r'C:\Users\paulk\OneDrive - University of Toronto\engsci 1t9\python_scripts\ConvolutionalNeuralNetworks\model_save.h5', overwrite=True)
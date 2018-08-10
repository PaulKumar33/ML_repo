from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers


def vectorizeSequences(seq, dimension = 10000):
    results = np.zeros((len(seq), dimension))
    for i, sequence in enumerate(seq):
        results[i, sequence] = 1
    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#partitioning the data
x_train = vectorizeSequences(train_data)
x_tes = vectorizeSequences(test_data)

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

print(x_train)
print(train_data)

model = models.Sequential()
model.add(layers.Dense(16,activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics=['accuracy'])

#now for training the model
history = model.fit(partial_x_train, partial_y_train,
                    epochs=20, batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,21)
plt.plot(epochs, loss_values, 'bo', label="training loss")
plt.plot(epochs, val_loss_values, 'b', label="validation loss")
plt.title("Training and validation loss")
plt.xlabel("epochs")
plt.ylabel('Loss')
plt.legend()
plt.show()

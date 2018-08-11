from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
import numpy as np
import time
import matplotlib.pyplot as plt

try:
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
except Exception as e:
    print(e)

def vectorizeSeq(seq, dimension=10000):
    #here we are creating a matrix with len(seq) rows and and 10000 cols
    result = np.zeros((len(seq), dimension))
    for i, seqs in enumerate(seq):
        result[i, seqs] = 1
    return result

x_train = vectorizeSeq(train_data)
x_test = vectorizeSeq(test_data)

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
#since our daya contains 46 layers (ie a 1x46 vector), we need an output
#for 46 different probabilities
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))

loss = history.history['loss']
val_loss = history.history['val_loss']
acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(loss)+1)
print(epochs)

plt.plot(epochs, loss, 'bo', label="training loss")
plt.plot(epochs, val_loss, 'go', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
plt.plot(epochs, acc, 'bo', label="training acc")
plt.plot(epochs, val_acc, 'go', label='Validation acc')
plt.title("Training and Validation Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
print(results)

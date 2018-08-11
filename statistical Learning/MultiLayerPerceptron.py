import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import csv
mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

'''this is an example of a simple neural network. Use as reference for further 
developement'''

'''print(type(mnist))
print(mnist.train.images)
print(mnist.train.images[2].reshape(28,28))
sample = mnist.train.images[2].reshape(28,28)
plt.imshow(sample, cmap='Greys')
plt.show()'''

learningRate = 0.001
trainingEpocs = 5
batchSize = 100

#network parameters
nClasses = 10
nSamples = mnist.train.num_examples
nInput = 784
nhidden1 = 256
nhidden2 = 256
nhidden3 = 256

#using the atom optimizer for the loss function
def MultilayerPerception(x,weights,biases):
    '''
    this is a neural network composed of 2 hidden layers and an ouput layer
    x: PLaceholder for data
    weights: Dict of weights
    biases: Dict of bias values
    '''
    #X*W + B
    layer1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    #Func(X*W+B) = Relu -> f(x) = max(0,x)
    layer1=tf.nn.relu(layer1)

    #seocnd hidden
    layer2=tf.add(tf.matmul(layer1,weights['h2']),biases['b2'])
    layer2=tf.nn.relu(layer2)

    layer3=tf.add(tf.matmul(layer2,weights['h3']),biases['b3'])
    layer3=tf.nn.relu(layer3)

    #output layer
    outputLayer = tf.matmul(layer3, weights['out']) + biases['out']

    return outputLayer

def WriteTxtFile(file, dict):
    '''
    takes a text file and dictionary as
    input and writes to the text file
    '''
    f = open(file, 'w')
    for keys in dict:
        f.write(keys + ": ")
        for element in dict[keys]:
            f.write(str(element))
    #f.write(dict)
    f.close()

weights={
    'h1':tf.Variable(tf.random_normal([nInput, nhidden1])),
    'h2':tf.Variable(tf.random_normal([nhidden1,nhidden2])),
    'h3':tf.Variable(tf.random_normal([nhidden2,nhidden3])),
    'out':tf.Variable(tf.random_normal([nhidden3,nClasses]))
}

biases = {
    'b1':tf.Variable(tf.random_normal([nhidden1])),
    'b2':tf.Variable(tf.random_normal([nhidden2])),
    'b3':tf.Variable(tf.random_normal([nhidden3])),
    'out':tf.Variable(tf.random_normal([nClasses]))
}

x = tf.placeholder('float',[None,nInput])
y = tf.placeholder('float',[None,nClasses])

pred = MultilayerPerception(x,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

#training the model

sess = tf.InteractiveSession()
init = tf.initialize_all_variables()
sess.run(init)

#beginning the training
for epoch in range(trainingEpocs):
    #15loops
    avgCost = 0.0
    totalBatch = int(nSamples/batchSize)
    for i in range(totalBatch):
        batchX,batchY = mnist.train.next_batch(batchSize)
        _,c=sess.run([optimizer,cost],feed_dict={x:batchX,y:batchY})
        avgCost += c/totalBatch
    print("Epoch: {} cost{:.4f}".format(epoch+1,avgCost))
print("Model has completed {} Epochs of traing".format(trainingEpocs))

correctPred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
correctPred = tf.cast(correctPred,'float')
accuracy = tf.reduce_mean(correctPred)

#mnist.test.labels[0]
print(accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

result = sess.run(weights)
#attempting to unpack the result into a regular dictionary
resultDictionary={}
for keys in result:
    resultDictionary[keys]=result[keys]
print(result)
print('printing he result of the first hidden layer weigths....')
print('\n\n\n')
print(resultDictionary['h1'])
print("size of the weights for h1: " +str(resultDictionary['h1'].shape))
WriteTxtFile('weights', resultDictionary)


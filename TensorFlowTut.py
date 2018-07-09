'''this is a tutorial in tensor flow for learning neural networks'''
import tensorflow as tf
import time
import numpy as np

#defining a constant
x = tf.constant(100)

#creating a tensor flow session
sess = tf.Session()
print(sess.run(x))

#tensor flow operations
x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print("Operations with constants")
    print("Addition: ",sess.run(x+y))
    print("Multiplication: ",sess.run(x*y))

#now place holder operations
time.sleep(3)
x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
print(x)
print(y)
add = tf.add(x,y)
with tf.Session() as sess:
    print("Operations with placeholders")
    #we pass in the values as a dictionary, with the place
    #holders as the dictionary keys
    print("Addition: ",sess.run(add, feed_dict={x:20,y:30}))

#opertions with arrays and matrices
a = np.array([[5.0, 5.0]])
b = np.array([[2.0], [2.0]])
mat1 = tf.constant(a)
mat2 = tf.constant(b)
matrix_multi = tf.matmul(mat1,mat2)
print('Performing matrix operations....')
time.sleep(3)
with tf.Session() as sess:
    result = sess.run(matrix_multi)
    print("Multiplaction: ", result)


'''this library is a self implemented stats library for my own use. the purpose is to better my programming and logic skills as well as make proccesses easier'''
import numpy as np
import time

class VectorMath:
    def __init__(self):
        '''initialize the class'''

    def VectorMean(self, vector):
        sum = np.sum(vector[i] for i in range(len(vector)))
        mean = sum/len(vector)
        return mean

    def VectorVariance(self, vector):
        vectMean = self.VectorMean(vector)
        sum = np.sum((vector[i]-vectMean)**2 for i in range(len(vector)))
        variance = sum/len(vector)
        return variance

    def Activation(self, vector, activation):
        '''this method activates a vector passed into it using relu, sigmoid, or tanh '''
        if(activation == "relu"):
            #for i in range(len(vector)):
            vector = np.maximum(0, vector)
        if(activation == "tanh"):
            exponential = np.multiply(-1, vector)
            exponential = np.multiply(2, exponential)
            den = 1+(np.exp(exponential))
            tanh = 2/den + 1
            vector = tanh

        if(activation == "sigmoid"):
            print("implementing sigmoid activation")
            time.sleep(2)
            exponential = np.multiply(-1, vector)
            den = 1+(np.exp(exponential))
            sigmoid = 1/den
            #print(sigmoid)
            vector = sigmoid

        return vector



if __name__ == "__main__":
    vect = VectorMath()
    vectorEle = [1,2,-3,4,5,-6]
    print(vect.VectorMean(vector=vectorEle))
    print(vect.VectorVariance(vectorEle))
    print(np.var(vectorEle))
    print(vect.Activation(vectorEle, "tanh"))


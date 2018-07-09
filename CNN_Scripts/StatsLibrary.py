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

    def AvgPool(self, matrix, size, pooltype='max'):
        size = size
        matSize = np.shape(matrix)
        print(matSize)
        poolSize = (matSize[0]-size+1,matSize[0]-size+1)
        newMat = np.zeros(poolSize)
        print(newMat)
        if(pooltype=='max'):
            for index1 in range(len(matrix)-1):
                for index2 in range(len(matrix)-1):
                    subMatrix = matrix[index1:index1+size,index2:index2+size]
                    print(subMatrix)
                    maximum = np.max(subMatrix)
                    newMat[index1,index2] = maximum
        if(pooltype=='avg'):
            for index1 in range(len(matrix)-1):
                for index2 in range(len(matrix)-1):
                    subMatrix = matrix[index1:index1+size,index2:index2+size]
                    print(subMatrix)
                    maximum = np.mean(subMatrix)
                    newMat[index1,index2] = maximum
        return newMat


if __name__ == "__main__":
    vect = VectorMath()
    vectorEle = [1,2,-3,4,5,-6]
    matrix = np.matrix([[1,1,4,2,3],
                        [1,2,9,4,6],
                        [0,2,2,3,0],
                        [8,8,1,2,3],
                        [5,4,7,6,1]])
    print(vect.VectorMean(vector=vectorEle))
    print(vect.VectorVariance(vectorEle))
    print(np.var(vectorEle))
    print(vect.Activation(vectorEle, "tanh"))
    print(vect.AvgPool(matrix,2,'avg'))


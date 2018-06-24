'''numpy learning'''
import numpy as np
## creating vectorsm arrays and operators

vec = np.zeros(10)          #creates a vector of zeros
print(vec)

vec2 = np.ones(10, int)     #creates vector on ones
print(vec2)
print("\n")

vec3 = np.ones(10, complex) #creates complex numbers
print(vec3)

#we can also use operations

vec2 = vec2 + 2
print(vec2)

vec1 = np.random.rand(4)
vec2 = np.random.rand(4)
final = vec1+vec2

matrix = np.random.ranf((2,3))      #creating a mtrix
tensor = np.ones((10,10,10), float)
el = matrix[1, 0]
el2 = matrix[0, 2]                  #accessing matrix elements. recall python starts at 0

#matrices can be multiplied using the dot() method
m = np.random.ranf((2,3))
v = np.random.rand(3)
product = np.dot(m, v)
product2 = m*v


## data types
# np.dtype returns data type
mat = np.random.ranf((3,4))
max = mat.max()
max_col = mat.max(0)            #returns the max of each col
mean = max.mean()               #returns the mean
max_arg = mat.argmax()          #returns max arg indice
                                #.shape returns dim
                                #divmod 
vec = np.random.rand(6)
gvec = vec > 0.3
nz = gvec.nonzero()             #nonzero gives locations where the value isnt zero. note gvec is a ool and thus false reads zero and true reads 1. it is a good way to threshold

#few math function to consider
'''sin arcsin sinh arcsinh
cos arccos cosh arccosh
tan arctan tanh arctanh
exp log log2 log10
sqrt conjugate floor ceil'''

##using numpy to sort data
#this module explores the effects of argsort

vec = np.random.rand(7)
ag = vec.argsort()              #argsort sorts the arguments
vec_sort = vec[ag]

print("the vector is", vec)
print("\n")
print("while the result of argsort is: ", vec_sort)

##strings and working with bytes in numpy

a = (16384*np.random.rand(1)).astype(int)
a_string = a.tostring()
map(ord, a.tostring())                  #look into what this is later
map(ord, a.byteswap().tostring())

##more on matrices
M =  np.random.ranf((5,6))
M_trans = M.transpose()
#print("the shape of M: {0}. The Shape of M_trans: {1}", M.shape, M_trans.shape)


M.resize((5,2,3))
#print(M)
#print("M has now been resized as: {0}", M.shape)

V = np.random.rand(10)
V[::2]                          #takes every second element

print(V)
print("\n")
n = [4, 1, 9, 6]
print("this is V[n]:{0} ", V[n])
print("\n")

Reorder = V[n]
V[n] = -1, -3, 6, 10            #changes the elements indiced in n
print(V)


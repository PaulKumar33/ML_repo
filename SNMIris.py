
'''this is the udemy project for learning suport vector machines'''
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)

iris = sb.load_dataset('iris')
print(iris)
print("\n print the features of the data ->"
      "\n Sepal Length: " +str(iris['sepal_length'])+
      "\n sepal_width: "+str(iris['sepal_width'])+
      "\n petal_length: " +str(iris['petal_width'])+
      "\n petal_width: " +str(iris['petal_width'])+
      "\n pecies: " + str(iris['species']))


#splitting and training the data
'''note the train_test_split() method takes the following
train_test_split(x,y,test_Size=,random_state=) ->
x -> is the x features
y -> label of the data
'''
dfx = pd.DataFrame(iris, columns=['sepal_length', 'sepal_width', 'petal_width', 'petal_length'])
print(dfx.info())
dfy = pd.DataFrame(iris, columns=['species'])
print(dfy.info())

dfx2 = iris.drop('species', axis = 1)
dfy2 = iris['species']


from sklearn.svm import SVC
from sklearn.metrics import classification_report
xtr, xt, ytr, yt = train_test_split(dfx, dfy, test_size=0.30, random_state=101)
xtr2, xt2, ytr2, yt2 = train_test_split(dfx2, dfy2, test_size=0.30, random_state=101)

model = SVC()
model.fit(xtr, ytr)
prediction = model.predict(xt)

model2 = SVC()
model2.fit(xtr2, ytr2)
prediction2 = model2.predict(xt2)

print("comparing the two methods of data splitting \n ")
print("\n method 1")
print(classification_report(yt, prediction))
print('\n method 2')
print(classification_report(yt2, prediction2))

'''now performing grid search methods'''
'''grid search we must input a list of paramters (c,gamma, kernel) and
let this run to find the optimal param'''

from sklearn.model_selection import GridSearchCV
param = {'C':[0.01, 0.1, 1, 10, 100], 'gamma': [0.0001,0.001,0.01,0.1,1],'kernel':['rbf']}
grid = GridSearchCV(SVC(), param,refit=True,verbose=10)
grid.fit(xtr, ytr)
print(grid.best_params_)
gridPredict = grid.predict(xt)
print(classification_report(yt, gridPredict))

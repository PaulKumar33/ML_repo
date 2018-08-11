import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('kyphosis.csv')
print(df.head())

'''sb.pairplot(df,hue='Kyphosis',palette='Set1')
plt.show()'''

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

xtr, xt, ytr, yt = train_test_split(X, y, test_size=0.30)

#implenting the sklearn decision tree model
dTree = DecisionTreeClassifier()
dTree.fit(xtr,ytr)

predictions = dTree.predict(xt)

print(classification_report(yt, predictions))
print(confusion_matrix(yt, predictions))

#------------------------------------------------------------------------#
###-------------implementing the visulization of the tree--------------###
#------------------------------------------------------------------------#

"""implementing random forests"""
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(xtr,ytr)
rfcPred = rfc.predict(xt)
print(confusion_matrix(yt,rfcPred))
print(classification_report(yt,rfcPred))


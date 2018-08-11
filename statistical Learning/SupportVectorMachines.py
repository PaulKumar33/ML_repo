import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

#use the breast cancer library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print(cancer.keys())
print(cancer['DESCR'])

dfFeat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
print(dfFeat.info())
dfTarget = pd.DataFrame(cancer['target'], columns=['Cancer'])

#splitting the dataset

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

xtr, xt, ytr, yt = train_test_split(dfFeat,np.ravel(dfTarget), test_size=0.30,random_state=101)
model = SVC()
model.fit(xtr, ytr)
prediction = model.predict(xt)
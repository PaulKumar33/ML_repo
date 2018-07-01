'''k nearest neighbours is essentially for data with no features/unkown features but does have target classes'''
'''the k nearest neighbours algorithm takes the nearest neighbours of the data point and predicts its label from surround data'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Classified_Data", index_col=0)
print(df.head())
#for element in df:
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))         #std scalar scales the values between 1 and 0
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_features = pd.DataFrame(scaled_features, columns=df.columns[:-1])
print(df_features.head())

xtr, xt, ytr, yt = train_test_split(scaled_features, df['TARGET CLASS'], test_size=0.3)
#looks for just one neighbour for classifying
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xtr, ytr)
pred = knn.predict(xt)

#evaluating the model
print(confusion_matrix(yt, pred))
print(classification_report(yt, pred))

#employing the elbow method to choose best k
error = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtr, ytr)
    pred_i = knn.predict(xt)
    error.append(np.mean(pred_i != yt))

plt.figure(figsize=(10,6))
plt.plot(range(1,40), error, color = 'blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize='10')
plt.title("Error rate vs K value")
plt.xlabel("K")
plt.ylabel("Error")
plt.show()



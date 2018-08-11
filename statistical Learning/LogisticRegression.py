'''learning logistic regression'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn as skl

def ImputeAge(cols):
    Age = cols[0]
    Pclass = cols[1]
    if(pd.isnull(Age)):
        if(Pclass == 1):
            return 37
        elif(Pclass == 2):
            return 29
        else:
            return 24
    else:
        return Age

train = pd.read_csv('titanic_train.csv')
print(train.head())
print(train.isnull())
sb.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

#must use the pyplot module to plot seaborn plots
plt.show()

#get a visual representation of survival
sb.countplot(x='Survived', data=train, hue='Sex')

'''Add more on exploratory data on age, spouses, siblings, gender, and class'''
plt.show()

#here we are passing the required data labels in and we
#are applying our function to fill in the missing data

train['Age'] = train[['Age', 'Pclass']].apply(ImputeAge,axis=1)
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
sb.heatmap(train.isnull(), yticklabels=False, cbar=False,cmap='viridis')
plt.show()

#now we are going to convert catagorical data into dummy variables
sex = pd.get_dummies(train["Sex"], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train = pd.concat([train,sex,embark], axis=1)
print("\n \n here is the head of the data \n\n")
print(train.head())

#now its time to build the logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

xtr, xt, ytr, yt = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.30,random_state=101)

logModel = LogisticRegression()
logModel.fit(xtr,ytr)

prediction = logModel.predict(xt)
print(classification_report(yt, prediction))



'''learning logistic regression'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import sklearn as skl

def ImputeAge(cols):
    age = cols[0]
    pclass = cols[1]
    if(pd.isnull(age)):
        if(pclass == 1):
            return 37
        elif(pclass == 2):
            return 29
        else:
            return 24
    else:
        return age

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



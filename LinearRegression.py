import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#import seaborn as sb
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("USA_Housing.csv")
df.head()
df.info()

print(df.columns)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df["Price"]
#we split our x and y data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
lm = LinearRegression()
#here we create a regresion model using sklearn and fit the test data
#to a line. from there we can get the intercepts and coefficients
lm.fit(X_train, y_train)
print(lm.intercept_)
print(lm.coef_)
cdf = pd.DataFrame(lm.coef_, X.columns, columns=["Coeff"])
print(cdf)

predictions = lm.predict(X_test)

'''evaluating the metrics'''
plt.scatter(y_test, predictions)
plt.show()

sb.distplot((y_test-predictions))

from sklearn import metrics
print(metrics.mean_absolute_error(y_test, predictions))
MSE = metrics.mean_squared_error(y_test, predictions)
RMSE = np.sqrt((MSE))
print(RMSE)

"""grabbing the boston housing dataset

from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())
"""



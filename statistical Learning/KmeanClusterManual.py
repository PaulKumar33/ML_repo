'''this is a manual implementation'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({
})

x=[]
y=[]
for i in range(26):
    assignmentX = np.random.randint(0,80)
    assignmentY = np.random.randint(0,80)
    x.append(assignmentX)
    y.append(assignmentY)

df['x']=x
df['y']=y

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

k = 3
'''assigning the centroids'''
centroids = {
    i+1:[np.random.randint(0,80), np.random.randint(0,80)]
    for i in range(k)
}

figure = plt.figure(figsize=(5,5))
plt.scatter(df['x'], df['y'], color='k')
colmap = {1:'r', 2:'g', 3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

def AssignCluster(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    print(df['closest'])
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

def update(centroids):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest']==i]['y'])
    return centroids

df = AssignCluster(df, centroids)
print(df.head())
centroids = update(centroids)

for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.scatter(df['x'], df['y'], color='k')
plt.show()

import copy
while True:
    closestCentroid = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = AssignCluster(df, centroids)
    if(closestCentroid.equals(df['closest'])):
        break

plt.scatter(df['x'], df['y'], color='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.show()


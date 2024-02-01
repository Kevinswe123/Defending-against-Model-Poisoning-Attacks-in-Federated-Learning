import numpy as np
from sklearn.cluster import KMeans
from mxnet import nd, autograd, gluon
from collections import Counter

data = np.array([0.254,0.254,0.264,0.77,0.57,0.55,0.31,0.53,0.59,0.317])
kmeans = KMeans(n_clusters=2).fit(data.reshape(-1,1))
kmeans.predict(data.reshape(-1,1))
print(kmeans.predict(data.reshape(-1,1)))
kmeans.cluster_centers_
print(kmeans.cluster_centers_)
g = Counter(kmeans.predict(data.reshape(-1,1)))

print(g)


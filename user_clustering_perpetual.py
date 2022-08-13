# https://mirror.xyz/barniker.eth/J6s7SYB4hUc90LXb7s0lmSNCIydrA2AzxHJep445_xw
# https://github.com/itzmestar/duneanalytics

from duneanalytics import DuneAnalytics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

# initialize client
dune = DuneAnalytics('rbun013', 'n9cJseTbd2wgSWu8i*%S')

# try to login
dune.login()

# fetch token
dune.fetch_auth_token()

# fetch query result id using query id
result_id = dune.query_result_id(query_id = 652723)

# fetch query result
data = dune.query_result(result_id)
result_list=data['data']['get_result_by_result_id']
result_list_clean=[e['data'] for e in result_list ]
d=pd.DataFrame(result_list_clean)

# Importing the dataset
X = d.iloc[:, [1,2,3]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s = 100, c = 'red', label = 'Cluster 1');
ax.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s = 100, c = 'blue', label = 'Cluster 2');
ax.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s = 100, c = 'green', label = 'Cluster 1');

ax.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])

ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s = 300, c = 'yellow', label = 'Centroids')
# ax.title('Clusters of users')
#ax.xlabel('xlabel')
#ax.ylabel('ylabel')
#ax.zlabel('zlabel')
# plt.show()
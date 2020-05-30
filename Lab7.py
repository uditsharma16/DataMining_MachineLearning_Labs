

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=150, 
                  n_features=2, 
                  centers=3, 
                  cluster_std=0.5, 
                  shuffle=True, 
                  random_state=0)


import matplotlib.pyplot as plt


from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, 
            init='random', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)

# # Organizing clusters as a hierarchical tree

# Clustering data using Agglomerative Clustering in bottom-up fashion


from sklearn.cluster import AgglomerativeClustering


ac = AgglomerativeClustering(n_clusters=3, 
                             affinity='euclidean', 
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)


# increase  the level of abstraction by decreasing the number of clusters 


ac = AgglomerativeClustering(n_clusters=2, 
                             affinity='euclidean', 
                             linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
plt.scatter(X[:, 0], X[:, 1])
plt.tight_layout()



# Compare K-means, hierarchical clustering and density based clustering:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))

km = KMeans(n_clusters=4, random_state=0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
            edgecolor='black',
            c='lightblue', marker='o', s=40, label='cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
            edgecolor='black',
            c='red', marker='s', s=40, label='cluster 2')
ax1.scatter(X[y_km == 2, 0], X[y_km == 2, 1],
            c='green', marker='*', s=40,
            edgecolor='black',
            label='cluster 3')
ax1.scatter(X[y_km == 3, 0], X[y_km == 3, 1],
            c='yellow', marker='x', s=40,
            edgecolor='black',
            label='cluster 4')
ax1.set_title('K-means clustering')

ac = AgglomerativeClustering(n_clusters=4,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c='lightblue',
            edgecolor='black',
            marker='o', s=40, label='cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1],c='green', marker='*',
            edgecolor='black',
             s=40, label='cluster 2')
ax2.scatter(X[y_ac == 2, 0], X[y_ac == 2, 1],c='red' ,marker='s'
            , s=40,
            edgecolor='black',
            label='cluster 3')
ax2.scatter(X[y_ac == 3, 0], X[y_ac == 3, 1],
            c='yellow', marker='x', s=40,
            edgecolor='black',
            label='cluster 4')
plt.tight_layout()
plt.show()


# Now apply the DBSCAN clusterer with the lower bound value of min_samples 
# discussed in the lecture. The eps parameter is harder to set; in this dataset 
# the default value of 0.5 did not discover two clusters

# In practice, 
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=.115, min_samples=4, metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1],
            c='lightblue', marker='o', s=40,
            edgecolor='black', 
            label='cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1],
            c='red', marker='s', s=40,
            edgecolor='black', 
            label='cluster 2')
plt.scatter(X[y_db == 2, 0], X[y_db == 2, 1],c='green', marker='*',
            edgecolor='black',
             s=40, label='cluster 3')
plt.scatter(X[y_db == 3, 0], X[y_db == 3, 1],
            c='yellow', marker='x', s=40,
            edgecolor='black',
            label='cluster 4')
plt.scatter(X[y_db == 4, 0], X[y_db == 4, 1],
            c='orange', marker='x', s=40,
            edgecolor='black',
            label='cluster 5')
plt.scatter(X[y_db == 5, 0], X[y_db == 5, 1],
            c='grey', marker='x', s=40,
            edgecolor='black',
            label='cluster 6')
plt.scatter(X[y_db == 6, 0], X[y_db == 6, 1],
            c='pink', marker='x', s=40,
            edgecolor='black',
            label='cluster 7')
plt.legend()
plt.tight_layout()
plt.show()







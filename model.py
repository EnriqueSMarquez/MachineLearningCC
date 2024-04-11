import numpy as np
from sklearn.cluster import KMeans

class CustomKMeans:
    def __init__(self, nb_clusters):
        self.nb_clusters = nb_clusters
        self.current_iteration = 0
        self.centroids = None

    def next_iteration(self, X):
        kmeans = KMeans(
            max_iter=1,
            n_init=1,
            init=(self.centroids if self.centroids is not None else 'k-means++'),
            n_clusters=self.nb_clusters,
            random_state=7)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_

        self.current_iteration += 1
        print(f'Finished iteration {self.current_iteration}')
        return self.centroids, kmeans.labels_
    
    def initialise(self, nb_clusters):
        self.current_iteration = 0
        self.centroids = np.random.uniform(low=-10, high=10, size=(nb_clusters, 2))

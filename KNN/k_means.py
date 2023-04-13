import numpy as np

class KMeans:
    def __init__(self, k, max_iter):
        self.k = k
        self.max = max_iter

    def initialize(self, centroids):
        # starting point (instead of random)
        self.centroids = centroids

    def wcss(self): # if i use func on fit twice i get 29 (bob)
        sum = 0
        for i in self.D.keys():
            sum += np.sum((np.linalg.norm(self.X[self.D[i]] - self.centroids[i], axis=1))**2)
        return sum

    def close_to_cen(self, row, centroids): #get row(point) and cento and return the index of closest cento
        distances = []
        for cent in centroids:
            dist = np.linalg.norm(row - cent)# (np.sum(np.sqrt((row - cent)**2)))
            distances.append(dist)          #add all dist to a list and then choose the index of lowest
        closest_centroid_id = np.argmin(distances)
        return closest_centroid_id

    def fit(self, X_train):
        self.X = X_train
        old_centroids = np.zeros(self.centroids.shape)
        for _ in range(self.max):
            D = {i: [] for i in range(self.k)}  #dict of emptys lists, as the numbers of clusters
            # Adjust clusters
            for i, row in enumerate(X_train):    #for each row(point) check with func the index of closest centroaid
                idx = self.close_to_cen(row, self.centroids)
                D[idx].append(i)                #connect between indx of cento t index of row(point)
            # Calculate Centroids
            old_centroids = self.centroids      #for checking at the end
            for k in range(self.k):                #for each key(cento) take the avrage(mean) of coordinates and replace
                new_c = np.mean(X_train[D[k]], axis=0)
                self.centroids = np.vstack((self.centroids[:k], new_c, self.centroids[k+1:]))
            # Check if changed
            if np.all(old_centroids == self.centroids):
                break
        self.D = D
        return {i: self.centroids[i] for i in range(self.k)}  #keys are indx of cento connect to suit cento(value)

    def predict(self, X):
        label_of_points = []
        for row in X:   #for each row(point) check with the func what index cento is bleong to
            rel_centroid = self.close_to_cen(row, self.centroids)
            label_of_points.append(rel_centroid)
        return np.array(label_of_points)    #return list of all index by order


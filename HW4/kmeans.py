import numpy as np
from itertools import combinations

#Distance function
def distance(x,y):
  return np.sqrt(((x - y)**2).sum(axis=1))

#Another distance function for 1d arrays
def distance_1d(x,y):
  return np.linalg.norm(x - y)

#Cluster makeing function, based on distance to centroids
def make_clusters(centroids, data, size, num_vals, K):
  clusters = dict.fromkeys(range(K), np.empty((0, num_vals)))
  for i in range(size):
    closest = closest_centroid(data[i,:-1], centroids)
    clusters[closest] = np.append(clusters[closest], [data[i]], axis=0)
  return clusters

#Determine closest centre
def closest_centroid(p, c):
  return np.argmin(distance(p,c), axis=0)

# Determine the old and new clusters are the same
def clusters_same(old, new):
  for o,n in zip(old.values(),new.values()):
    o_size = np.size(o)
    n_size = np.size(n)
    if n_size != o_size or not np.array_equal(o,n):
      return False
  return True

# Mean Squared Error function
def mse(clusters, centroids):
  errors = []
  for cluster, centroid in zip(clusters.values(), centroids):
    error = np.sum(distance(cluster[:,:-1], centroid)**2) / (np.size(cluster, axis=0) - 1)
    errors = np.append(errors, error)
  return errors

# Average Mean Squared Error function
def average_mse(clusters, centroids, K):
  return np.sum(mse(clusters, centroids)) / K

# Mean Squared Separation function
def mss(centroids, K):
  all_pairs = list(combinations(range(K), 2))
  distances = []
  for i,j in all_pairs:
    dist = distance_1d(centroids[i], centroids[j]) ** 2
    distances = np.append(distances, dist)
  return np.sum(distances) / (K * (K - 1) / 2)

# Entropy function
def entropy(cluster):
  instances = cluster[:,-1]
  _, count = np.unique(instances, return_counts=True)
  total = np.size(instances)
  count = count / total
  return -np.sum(count * np.log2(count))

# Mean Entropy function
def mean_entropy(clusters, total):
  entropies = np.array([entropy(cluster) for cluster in clusters.values()])
  probability = np.array([np.size(cluster, axis=0) / total for cluster in clusters.values()])
  return np.sum(probability * entropies)

# Classify associating digits function
def classify(clusters):
  classes = []
  for cluster in clusters.values():
    actuals = cluster[:,-1].astype(int)
    counts = np.bincount(actuals)
    most = np.argmax(counts)
    ties = np.argwhere(counts == most).flatten()
    if np.size(ties) > 1:
      most = np.random.choice(ties)
    classes.append(most)
  return classes

# Run function
# Creates the centroids, returns them and the average MSE
def run(data, K):
  # Create clusters from randomized array values
  data_rand = data.copy()
  np.random.shuffle(data_rand)

  data_size = np.size(data, axis=0)

  num_vals = np.size(data, axis=1)

  clusters = dict.fromkeys(range(K), np.empty((0, num_vals)))
  choice = 0
  for i in range(num_vals):
    clusters[choice] = np.append(clusters[choice], [data_rand[i]], axis=0)
    choice = (choice+1) % K

  # Create centroids, re-cluster, repeat until convergence
  while True:
    centroids = np.array([np.mean(cluster[:,:-1], axis=0) for cluster in clusters.values()])
    new_clusters = make_clusters(centroids, data, data_size, num_vals, K)
    if clusters_same(clusters, new_clusters):
      break
    else:
      clusters = new_clusters

  return centroids, average_mse(clusters, centroids, K)

import kmeans as k
import numpy as np
import PIL
from PIL import Image

# Detemine amount of clusters, K
K = 10

# Read/gain data from train/test files
opt_train = np.loadtxt("optdigits.train", delimiter=",")

opt_test = np.loadtxt("optdigits.test", delimiter=",", dtype=int)
opt_test_actuals = opt_test[:,-1]
test_size = np.size(opt_test_actuals)

attributes = np.size(opt_train, axis=1) - 1

data_size = np.size(opt_train, axis=0)

num_vals = np.size(opt_train, axis=1)

# determine how many times to generate the clusters
runs = 5

run_centroids = np.empty((runs, K, attributes))
run_amse = np.empty(runs)

# loop runs times, generating centroids and average MSE
for i in range(runs):
  run_centroids[i] , run_amse[i] = k.run(opt_train, K)

# Best average MSE, meaning lowest
best = np.argmin(run_amse)

# Centroids correspong to best average MSE
centroids = run_centroids[best]

# Generate the clusters and classify the cluster with 0-9
clusters = k.make_clusters(centroids, opt_train, data_size, num_vals, K)
classifiers = k.classify(clusters)

# Print associated digits
print("Associated digits: ", classifiers)

# print average MSE, MSS and mean entropy
mss = k.mss(centroids, K)
mean_entropy = k.mean_entropy(clusters, data_size)
print("Average MSE: %.2f" % best)
print("MSS: %.2f" % mss)
print("Mean entropy: %.2f" % mean_entropy)

# Generate confusion matrix
confusion = np.zeros((K,K))
predictions = []

for i in range(test_size):
  closest = k.closest_centroid(opt_test[i,:-1], centroids)
  ties = np.argwhere(k.distance(opt_test[i,:-1], centroids) == k.distance_1d(opt_test[i,:-1], centroids[closest])).flatten()
  if np.size(ties) > 1:
    closest = np.random.choice(ties)
  prediction = classifiers[closest]
  predictions = np.append(predictions, prediction)
  confusion[opt_test_actuals[i], prediction] += 1

# Print confusion matrix
print("Confusion matrix:")
print(confusion)
# Print accuracy
print("Accuracy: %.2f %%" % (np.sum(confusion.diagonal()) / test_size * 100))

file_names = ["cluster"] * K
for i in range(K):
  file_names[i] = file_names[i] + str(i) + ".png"
  matrix = centroids[i].copy().reshape((8,8))
  image = Image.fromarray(matrix, "L")
  image.save(file_names[i])


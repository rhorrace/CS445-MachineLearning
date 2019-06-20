from sklearn import svm
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

# Build training/test data function
def build():
  data = np.genfromtxt("spambase.data", delimiter=",")
  # Split between positives and negatives
  positives, negatives = data[data[:,-1] == 1,:], data[data[:,-1] == 0,:]

# Shuffle them
  np.random.shuffle(positives)
  np.random.shuffle(negatives)

# Split positives/negatives into positive/negative train/test
  p_size, n_size = np.size(positives, axis=0), np.size(negatives, axis=0)
  p_mid, n_mid = (p_size // 2) + 1, (n_size // 2) + 1
  p_train, p_test = positives[:p_mid, :], positives[p_mid:, :]
  n_train, n_test = negatives[:n_mid, :], negatives[n_mid:, :]

  # Combine to make training test
  train, test = np.concatenate((p_train, n_train), axis=0), np.concatenate((p_test, n_test), axis=0)

  # Shuffle training
  np.random.shuffle(train)
  np.random.shuffle(test)
  return train, test

# Standardize function
def standardize(X, mean, std):
  return (X - mean) / std

# Build training/test data
training, testing = build()

# Split training/test attributes and actuals
train_X = training[:,:-1]
train_Y = training[:,-1]
test_X = testing[:,:-1]
test_Y = testing[:,-1]

# Training mean and standard deviation
train_mean, train_std = np.mean(train_X, axis=0), np.std(train_X, axis=0)

# Scale training and testing data
# using training mean and std
n = np.size(train_std)
for i in range(n):
  if train_std[i] == 0:
    train_X[:, i] = 0
    test_X[:, i] = 0
  else:
    train_X[:,i] = standardize(train_X[:,i], train_mean[i], train_std[i])
    test_X[:,i] = standardize(test_X[:,i], train_mean[i], train_std[i])

# Generate SVM
classifier = svm.SVC(kernel="linear")

# Fit training data
classifier.fit(train_X, train_Y)

# Get predictions
test_score = classifier.predict(test_X)

# Get accuracy, precision, and recall
accuracy = classifier.score(test_X, test_Y)
precision = metrics.precision_score(test_Y, test_score)
recall = metrics.recall_score(test_Y, test_score)

# Print them
print("Accuracy:\t", accuracy)
print("Precision:\t", precision)
print("Recall:\t\t", recall)

# Get FPR, tPR, and threshold(s)
fpr, tpr, threshold = metrics.roc_curve(test_Y, test_score)

# Get AUC
roc_auc = metrics.auc(fpr, tpr)

# Create ROC curve
plt.title("ROC curve")
plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
plt.legend(loc = "lower right")
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("TPR")
plt.xlabel("FPR")

# Generate threshold plot on same ROC curve
ax2 = plt.gca().twinx()
plt.plot(fpr, threshold, markeredgecolor="r", linestyle="dashed", color="r")
ax2.set_ylabel("Threshold", color="r")
ax2.set_ylim([threshold[-1], threshold[0]])
ax2.set_xlim([fpr[0], fpr[-1]])

# Display plot
plt.show()

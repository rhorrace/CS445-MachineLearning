# Robert Horrace
# 967743553

import numpy as np
import network as N

# Convert actual to output targets
def identity(n):
  matrix = np.identity(n, dtype = float)
  matrix = np.where(matrix == 1, 0.9,0.1)
  return matrix
# #nd convert

# Begin Assigment 2 program

#Epochs
epochs = 50

# Number of inputs
num_inputs = 785

# Number of Hidden/Output perceptrons
H_n = 20
O_n = 10

network = N.Network(num_inputs, H_n, O_n)
# Read from training data csv
training = np.genfromtxt('mnist_train.csv', dtype=int, delimiter=',')

# Read from testing data csv
testing = np.genfromtxt('mnist_test.csv', dtype=int, delimiter=',')

# number of training inputs
num_training = np.size(training,axis=0)

# Number of testing outputs
num_testing = np.size(testing,axis=0)

# Shuffle training data by row
np.random.shuffle(training)

# Capture actual values
train_actuals = np.array(training[:,0], dtype=int)
training = np.delete(training,0,1)
test_actuals = np.array(testing[:,0], dtype = int)
testing = np.delete(testing,0,1)

O_t = identity(O_n)

# Scale data for smaller weights
training = np.divide(training, 255.0)
testing = np.divide(testing, 255.0)

# Epoch 0
train_predictions = network.predict(training)
test_predictions = network.predict(testing)

print("Epoch 0):")

# Display training accuracy
print("Training data Accuracy: %.2f %%" % (np.sum(train_predictions == train_actuals) / num_training * 100.0))

# Display testing accuracy
print("Testing data Accuracy: %.2f %%" % (np.sum(test_predictions == test_actuals) / num_testing * 100.0))


# run for i epochs
for i in range(epochs):
  print("Epoch %d):" % (i+1))

  # Training loop
  for i in range(num_training):
    t = O_t[train_actuals[i]]
    network.back_prop(training[i,:], t)
  # End training loop

  # Get prediction of training/testing
  train_predictions = network.predict(training)
  test_predictions = network.predict(testing)

  # Display training accuracy
  print("Training data Accuracy: %.2f %%" % (np.sum(train_predictions == train_actuals) / num_training * 100.0))

  # Display testing accuracy
  print("Testing data Accuracy: %.2f %%" % (np.sum(test_predictions == test_actuals) / num_testing * 100.0))
# End epoch loop

# Confusion matrix
confusion = np.zeros(100, dtype=int).reshape((10,10))

# Develop confusion matrix with testing data
for i in range(num_testing):
  confusion[test_actuals[i],test_predictions[i]] += 1

# Display confusion matrix
print("Confusion matrix:\n", confusion)

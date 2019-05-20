import numpy as np
import perceptron as P

# Variables used
# number of epochs 0-50
epochs = 1

# Initialized perceptrons: 10 total
perceptrons = [P.Perceptron(785,0.1) for _ in range(10)]
for i in range(10):
  perceptrons[i] = P.Perceptron(785,0.1)

# Training data and Testing data from csv files
training = np.genfromtxt('mnist_train.csv', dtype=int, delimiter=',')
testing = np.genfromtxt('mnist_test.csv', dtype=int, delimiter=',')

# Shuffle the training data
np.random.shuffle(training)

# Number of training/testing data
num_training = len(training)
num_testing = len(testing)

# Identity matrix of target outputs wanted
target_outputs = np.identity(10, dtype=int)

# Target values of training/testing data
train_targets = list(training[:,0])
test_targets = list(testing[:,0])

# Adjusting data x / 255
training[:,0] = 1
training[1:,:] = training[1:,:] / 255.0
testing[:,0] = 1
testing[1:,:] = testing[1:,:] / 255.0

# Epoch training:
#  Go through Training data
#  Calculate accuracy of predictions
#  Go through Testing data
#  Calculate accuracy of predictions
#  Train perceptions
for e in range(epochs):
  outputs = np.empty((0,10), dtype=int)
  guesses = []
  print("epoch %d:" % e)
  # prediction for loop
  for i in training:
    guess = np.argmax([p.compute(i) for p in perceptrons])
    guesses = np.append(guesses, [guess])
    output = [p.predict() for p in perceptrons]
    outputs = np.append(outputs, [output], axis=0)
  #end

  # Compute training accuracy
  print("\tTraining Accuracy: %.2f" % ((np.sum(guesses == train_targets) / num_training) * 100.0))

  outputs = np.empty((0,10), dtype=int)
  guesses = []

  # Go through testing data
  for i in testing:
    guess = np.argmax([p.compute(i) for p in perceptrons])
    guesses = np.append(guesses, [guess])
    output = [p.predict() for p in perceptrons]
    outputs = np.append(outputs, [output], axis=0)
  # end testing data predictions for loop

  # Compute testing accuracy
  print("\tTesting Accuracy: %.2f" % ((np.sum(guesses == test_targets) / num_testing) * 100.0))

  # Stop training if # of epochs passed
  if e == epochs-1:
    break

  # Training using all training data
  for i,out,target in zip(training, outputs, train_targets):
    for p,y,t in zip(perceptrons, out, target_outputs[target]):
        p.train(i,y,t)
    # end training through perceptrons for loop
  # end training for loop
# end epoch for loop

# Create confusion matrix for testing data
confusion = np.zeros((10,10), dtype=int)
for i,t in zip(testing, test_targets):
  guess = np.argmax([p.compute(i) for p in perceptrons])
  confusion[t,guess] += 1

# Display confusion matrix
print("Testing confusion matrix:")
print("# of outputs: ", np.sum(confusion))
print(confusion)

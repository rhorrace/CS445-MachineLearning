import numpy as np

def convert(x):
  identities = np.empty((0,10),int)
  for i in x:
    identity = np.zeros((1,10),int)
    identity[0,i] = 1
    identities = np.append(identities, identity, axis=0)
  return identities


learn_rate = [0.1]

epochs = 51

perceptrons = np.random.uniform(-0.5,0.5,785*10).reshape((785,10))

training = np.genfromtxt('mnist_train.csv', dtype=int,delimiter=',')
testing = np.genfromtxt('mnist_test.csv', dtype=int,delimiter=',')

np.random.shuffle(training)

num_training = len(training)
num_testing = len(testing)

train_targets = np.array(training[:,0], dtype=int)
test_targets = np.array(testing[:,0], dtype=int)


train_actuals = convert(train_targets)
test_actuals = convert(test_targets)

training = np.divide(training, 255.0)
training[:,0] = 1.0
testing = np.divide(testing, 255.0)
testing[:,0] = 1.0

for i in range(epochs):
  train_guesses = np.dot(training, perceptrons)
  train_predictions = np.argmax(train_guesses, axis=1)
  print("Training data Accuracy: %.2f" % (np.sum(train_predictions == train_targets) / num_training * 100.0))

  test_guesses = np.dot(testing, perceptrons)
  test_predictions = np.argmax(test_guesses, axis=1)
  print("Testing data Accuracy: %.2f %%" % (np.sum(test_predictions == test_targets) / num_testing * 100.0))

  train_ys = np.where(train_guesses > 0.0, 1, 0)

  errors = np.subtract(train_actuals, train_ys)
  errors = np.multiply(errors, learn_rate)

  delta_w = np.dot(np.transpose(training), errors)
  perceptrons = np.add(perceptrons, delta_w)


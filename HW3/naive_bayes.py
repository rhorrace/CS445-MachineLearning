import numpy as np
import classifier as c

# naive bayes classifier function
def naive_bayes(training_file, test_file):

  # Training and testing data
  train = np.genfromtxt(training_file)
  test = np.genfromtxt(test_file)

  # Gather data for training
  train_t = train[:,-1].astype(int)
  train_t_max = np.amax(train_t)
  train_total = np.size(train_t)
  train = train[:,:-1]
  train_d_size = np.size(train,axis=1)

  # Gather data for testing
  test_t = test[:,-1].astype(int)
  test_total = np.size(test_t)
  test = test[:,:-1]
  test_d_size = np.size(test,axis=1)

  # Create classifiers
  classifiers = c.Classifier(train_t_max, train_d_size)

  # Get total prpbabilities
  total_prob = np.zeros(train_t_max)
  for i in range(train_t_max):
    total_prob[i] = np.sum(train_t == (i+1)) / train_total

  # Place training data in classifiers
  for i in range(train_total):
    classifiers.adjust(train[i,:], train_t[i])

  # Display classifiers after training
  print("Training phase:")
  classifiers.display()

  # Testing phase
  print("Testing phase:")
  accuracies = []
  for i in range(test_total):
    # Get predicted class, probability, and classes with same probability
    predicted, probability, classes = classifiers.predict(test_t[i], total_prob)
    # Compute accuracy
    accuracy = 0
    if(predicted-1 in classes):
      if(classes.size == 1):
        if(predicted == test_t[i]):
          accuracy = 1
        else:
          accuracy = 0
      else:
        accuracy = 1/classes.size
    else:
      accuracy = 0
    # Display test ID, predicted class, probability, true value, and prediction accuracy
    print("ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f" % (i+1, predicted, probability, test_t[i], accuracy))
    accuracies = np.append(accuracies, accuracy)

  # Print  classification accuracy
  print("classification accuracy=%6.4f" % np.mean(accuracies))
# end naive_bayes

# Execution script
print("enter training file:",end=" ")
training_file = input()
print(training_file)
print("enter testing file:",end=" ")
testing_file = input()
print(testing_file)

naive_bayes(training_file, testing_file)

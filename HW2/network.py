import numpy as np

class Network(object):
  # Initialize function, takes in number of inputs, hidden/output units
  def __init__(self, num_inputs, H_n, O_n):
    # Hidden matrix (inputs x hidden units)
    self.H = np.random.uniform(-0.05,0.05,num_inputs*H_n).reshape((num_inputs,H_n))
    # Hidden dw matrix (inputs x num hidden units)
    self.H_dw = np.zeros((num_inputs,H_n))
    # Number of hidden units
    self.H_n = H_n
    # Output matrix (Hidden+1 x num output units)
    self.O = np.random.uniform(-0.05,0.05,(H_n+1)*O_n).reshape((H_n+1,O_n))
    # Output dw matrix (hidden+1 x )
    self.O_dw = np.zeros((H_n+1)*O_n).reshape((H_n+1,O_n))
    # Number of output units
    self.O_n = O_n
    # Momentum
    self.alpha = 0.9
    # Learn rate
    self.learn = 0.1
  # end __init__

  # Sigmoid function
  def sigmoid(self, X):
    return 1 / (1 + np.exp(-X))
  # End sigmoid

  # Sigmoid derivative function
  def sigmoid_prime(self, X):
    return X * (1 - X)
  # End sigmoid_prime

  # predict function
  def predict(self, X):
    # Input w/bias
    H_i = np.insert(X, 0, 1, axis=1)

    # Hidden unit guesses (sigmoid(inputs * Hidden))
    H_g = self.sigmoid(np.dot(H_i,self.H))

    # Hidden guesses w/bias
    O_i = np.insert(H_g, 0, 1, axis=1)

    # Output unit guesses (sigmoid(O inputs * outputs))
    O_g = self.sigmoid(np.dot(O_i,self.O))

    # Return max value of each input
    return np.argmax(O_g, axis=1)
  # End predict

  # Feed forward function
  def feed_forward(self, x):
    # Add bias to inputs
    H_i = np.insert(x, 0, 1)

    # Apply sigmoid function to x * perceptron weights
    H_g = self.sigmoid(np.dot(H_i,self.H))

    # Add bias to guesses for output perceptrons
    O_i = np.insert(H_g, 0, 1)

    # Sigmoid to H * output weights
    O_g = self.sigmoid(np.dot(O_i,self.O))

    # Return H guesses and O guesses for back prop
    return H_i, H_g, O_i, O_g
  # End feed_forward

  # Back propoagation function
  def back_prop(self, inputs, target):

    # Feed forward, getting inputs and outputs
    # of hidden and output layer
    H_i, H_g, O_i, O_g = self.feed_forward(inputs)

    # Errors of output units
    O_e = (target - O_g) * self.sigmoid_prime(O_g)

    # Errors of hidden units dependent on the output errors
    H_e = self.sigmoid_prime(O_i[1:]) * np.sum(np.sum(self.O[1:,:] * O_e, axis=1))

    O_i = np.tile(O_i, (O_e.size, 1)).T

    # Update ouputs dw
    #self.O_dw = (self.learn * O_e * O_i) + (self.alpha * self.O_dw)
    self.O_dw = (self.learn * O_e * O_i)

    # Update outputs weights
    self.O += self.O_dw

    H_i = np.tile(H_i, (H_e.size, 1)).T

    # Update hidden dw
    #self.H_dw = (self.learn * H_e * H_i) + (self.alpha * self.H_dw)
    self.H_dw = (self.learn * H_e * H_i)

    # Update hidden weights
    self.H += self.H_dw
  # End back_prop
# End Network

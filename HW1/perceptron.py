import numpy as np
import random as rand

# Horrace, Robert
# 967743553
# CS445

# This is the Perceptron class.
class Perceptron(object):

  # initialize function for Perceptron
  # Perceptron will have:
  #   Amount of epochs
  #   A learning rate
  #   Weights for the inputs
  def __init__(self, num_ins=1, learn_rate=0.1):
    self.computed = 0.0
    self.learn_rate = learn_rate
    self.n = num_ins
    self.W = np.random.uniform(-0.5,0.5,(num_ins))
  # end __init__

  # compute function
  def compute(self, X):
    self.computed = np.dot(X, self.W)
    return self.computed
  # end compute

  # predict/activate function
  def predict(self):
    return 1 if self.computed > 0 else 0
  # end predict

  # training function
  def train(self, X, y, t):
    errors = np.empty(self.n)
    errors.fill((t - y) * self.learn_rate)
    adjusted = np.multiply(errors,X)
    self.W = np.add(self.W, adjusted)
  # end train
#end Perceptron class

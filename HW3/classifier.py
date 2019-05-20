import numpy as np
import table as T

# Classifier class
class Classifier:
  # Initialize
  def __init__(self, num_classes, data_size):
    self.size = num_classes
    self.classes = np.zeros((num_classes,data_size))
    self.tables = [T.Table(data_size) for _ in range(self.size)]

  # Adjust table function
  def adjust(self, x, i):
    self.tables[i-1].add(x)
    self.tables[i-1].compute()

  # predict class, maximum probability, and max probability classes function
  def predict(self, x, p):
    for i in range(self.size):
      self.classes[i,:] = self.N(x, self.tables[i].mean, self.tables[i].std)
    probs = np.product(self.classes, axis=1) * p
    max_prob = np.amax(probs)
    max_args = np.argwhere(probs == max_prob).flatten()
    c = 0
    # If one max probability exist
    if(max_args.size == 1):
      c = max_args[0] + 1
    else:
      c = np.random.choice(max_args) + 1
    return c, max_prob, max_args

  # N function
  def N(self, x, mean, std):
    N1 = 1 / (np.sqrt(2 * np.pi) * std)
    N2 = np.exp(- (np.square(x - mean) / (2 * np.square(std))))
    N = N1*N2
    return N

  # Display classes and their tables
  def display(self):
    for i in range(self.size):
      self.tables[i].display(i+1)

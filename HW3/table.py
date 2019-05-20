import numpy as np

# Table class for data
class Table:
  # Initialize function
  def __init__(self, data_size):
    self.size = data_size
    self.data = np.empty((0,self.size))
    self.mean = np.zeros(self.size)
    self.std = np.full(self.size, 0.01)

  # Add data to Table
  def add(self, x):
    self.data = np.append(self.data, [x], axis=0)

  # compute mean and standard deviation
  def compute(self):
    if(self.data.size != 0):
      self.mean = np.mean(self.data,axis=0, dtype=float)
    self.std = np.std(self.data, axis=0, dtype=float)
    self.std = np.where(self.std < 0.01, 0.01, self.std)

  # Display attribute #, mean, and standard deviation
  def display(self, c):
    for i in range(self.size):
      print("Class %d, attribute %d, mean = %.2f, std = %.2f" % (c, i+1, self.mean[i], self.std[i]))

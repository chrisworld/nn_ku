#import tensorflow as tf
import numpy as np

class ClassifiedBatches():
  def __init__(self, examples, classes, batch_size, test=False):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.test = test
    #number of batches corresponding to batch_size
    self.batch_num = np.round(self.examples.shape[0] / batch_size) 
    #split all examples and classes
    self.examples = np.array_split(self.examples, self.batch_num)
    self.classes = np.array_split(self.classes, self.batch_num)
    if self.test == False:
    #split into validation and training set
       self.examples_train = self.examples[0:120]
       self.examples_validation = self.examples[120:]
       self.classes_train = self.classes[0:120]
       self.classes_validation = self.classes[120:]

    return
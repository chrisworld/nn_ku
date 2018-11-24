#import tensorflow as tf
import numpy as np

class ClassifiedBatches():
  def __init__(self, examples, classes, batch_size, is_test_set=False):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.is_test_set = is_test_set

    # Arrays
    self.examples_train = examples
    self.examples_validation = examples
    self.classes_train = classes
    self.classes_validation = classes

    # init Arrays if test set
    if self.is_test_set == False:
      #split into validation and training set
      self.examples_train = self.examples[0:120]
      self.examples_validation = self.examples[120:]
      self.classes_train = self.classes[0:120]
      self.classes_validation = self.classes[120:]

    # Batch List
    # number of batches corresponding to batch_size
    self.batch_num = np.round(self.examples.shape[0] / batch_size) 
    # split all examples and classes
    self.batch_examples_train = np.array_split(self.examples_train, self.batch_num)
    self.batch_classes_train = np.array_split(self.classes_train, self.batch_num) 


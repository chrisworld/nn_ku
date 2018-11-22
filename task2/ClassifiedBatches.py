import tensorflow as tf
import numpy as np

class ClassifiedBatches():
  def __init__(self, examples, classes, batch_size):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    #number of batches corresponding to batch_size
    self.batch_num = np.round(self.examples.shape[0] / batch_size) 
    self.batch_examples = np.array_split(self.examples, self.batch_num)
    self.batch_classes = np.array_split(self.classes, self.batch_num)





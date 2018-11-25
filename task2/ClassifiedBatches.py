#import tensorflow as tf
import numpy as np

class ClassifiedBatches():
  def __init__(self, examples, classes, batch_size, is_test_set=False, is_validation=True):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.is_test_set = is_test_set
    self.is_validation = is_validation

    if self.is_test_set == True:
        return

    # Arrays
    self.examples_train = examples
    self.examples_validation = examples
    self.classes_train = classes
    self.classes_validation = classes

    # return if no validation set is needed
    if self.is_validation == False:
      train_set_percent = 1.0
    else:
      train_set_percent = 0.7 

    # init Arrays if test set
    train_num =  int(round(train_set_percent*examples.shape[0]))
    print(train_num)
    
    #split into validation and training set
    self.examples_train = self.examples[0:(train_num)]
    self.examples_validation = self.examples[(train_num):]
    self.classes_train = self.classes[0:(train_num)]
    self.classes_validation = self.classes[(train_num):]

    # Batch List
    # number of batches corresponding to batch_size
    self.batch_num = np.ceil(self.examples_train.shape[0] / batch_size)

    # split all examples and classes
    self.batch_examples_train = np.array_split(self.examples_train, self.batch_num)
    self.batch_classes_train = np.array_split(self.classes_train, self.batch_num) 


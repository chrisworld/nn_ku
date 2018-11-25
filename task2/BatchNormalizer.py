import numpy as np
import tensorflow as tf
from ClassifiedBatches import ClassifiedBatches

class BatchNormalizer():
  def __init__(self, examples, classes=None, batch_size=40, shuffle=False, num_classes=26):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    #self.batch_num = np.round(self.examples.shape[0]/batch_size)  #number of batches corresponding to batch_size
    self.shuffle = shuffle
    self.num_classes = num_classes
    self.mean_features = np.mean(self.examples, axis=0)
    self.std_features = np.std(self.examples, axis=0)

  def shuffle_in_unison(self,examples,classes):
    assert len(examples) == len(classes)
    shuffled_a = np.empty(examples.shape, dtype=examples.dtype)
    shuffled_b = np.empty(classes.shape, dtype=classes.dtype)
    permutation = np.random.permutation(len(examples))
    for old_index, new_index in enumerate(permutation):
      shuffled_a[new_index] = examples[old_index]
      shuffled_b[new_index] = classes[old_index]
    return shuffled_a, shuffled_b

  def getMean(self):
    return self.mean_features

  def getStd(self):
    return self.std_features

  def getNormalized(self,examples, classes):
    if (self.shuffle==True):
      examples, classes = self.shuffle_in_unison(examples,classes)
    norm=(examples-(self.mean_features))/(self.std_features)
    print("normalized_data shape: ", norm.shape)
    print("normalized_data mean: ", norm.mean(axis=0)[0])
    print("normalized_data std: ", norm.std(axis=0)[0])
    return norm,classes

  def getBatches(self,examples,classes, test=False):
    norm,norm_classes = self.getNormalized(examples,classes)
    c_oh = tf.one_hot(norm_classes-1, self.num_classes)
    with tf.Session() as sess:
      self.classes_one_hot = sess.run(c_oh)
    cbatches = ClassifiedBatches(norm, self.classes_one_hot, self.batch_size, test)
    return cbatches


    
    #print("feature_score shape: ", feature_score.shape)
    #norm_examples =  
    #batch = ClassifiedBatches()
    return
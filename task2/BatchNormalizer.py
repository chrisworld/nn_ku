import numpy as np
import tensorflow as tf
from ClassifiedBatches import ClassifiedBatches

class BatchNormalizer():
  def __init__(self, examples, classes=None, batch_size=40, shuffle=False,num_classes=26):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.batch_num = np.round(self.examples.shape[0]/batch_size)  #number of batches corresponding to batch_size
    self.shuffle = shuffle
    self.num_classes = num_classes

  def getMean(self):
    mean_features = np.mean(self.examples, axis=0)
    print("mean_features shape: ", mean_features.shape)

    self.mean_features = mean_features
    return mean_features

  def getStd(self):
    std_features = np.std(self.examples, axis=0)
    print("std_features shape: ", std_features.shape)
    self.std_features = std_features
    return std_features

  def getNormalized(self):
    norm=(self.examples-(self.mean_features))/(self.std_features)
    print("normalized_data shape: ", norm.shape)
    print("normalized_data mean: ", norm.mean(axis=0)[0])
    print("normalized_data std: ", norm.std(axis=0)[0])
    self.norm = norm
    return norm

  def getBatches(self):
    c_oh = tf.one_hot(self.classes-1, self.num_classes)
    with tf.Session() as sess:
      self.classes_one_hot = sess.run(c_oh)

    cbatches = ClassifiedBatches(np.array_split(self.norm,self.batch_num),
                                 np.array_split(self.classes_one_hot, self.batch_num),
                                 batch_size = self.batch_size)
    print("single batch shape: ", cbatches.examples[0].shape)
    print("single class shape: ", cbatches.classes[0].shape)
    return cbatches


    
    #print("feature_score shape: ", feature_score.shape)
    #norm_examples =  
    #batch = ClassifiedBatches()
    return
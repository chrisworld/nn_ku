import numpy as np
from ClassifiedBatches import ClassifiedBatches

class BatchNormalizer():
  def __init__(self, examples, classes=None, batch_size=40, shuffle=False):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.shuffle = shuffle

  def getMean(self):
    mean_features = np.mean(self.examples, axis=0)
    print("mean_features shape: ", mean_features.shape)
    return mean_features

  def getStd(self):
    std_features = np.std(self.examples, axis=0)
    print("std_features shape: ", std_features.shape)
    return std_features

  def getNormalizedClassBatches(self):
    mean_features = self.getMean() 
    std_features = self.getStd()
    
    #print("feature_score shape: ", feature_score.shape)
    #norm_examples =  
    #batch = ClassifiedBatches()
    return
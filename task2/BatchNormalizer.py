import numpy as np
from ClassifiedBatches import ClassifiedBatches

class BatchNormalizer():
  def __init__(self, examples, classes=None, batch_size=40, shuffle=False):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.batch_num = np.round(self.examples.shape[0]/batch_size)
    self.shuffle = shuffle
    self.batch_list = np.array_split(self.examples,self.batch_num)
    self.batch_len = self.batch_list.__len__()
    self.num_feat = self.examples.shape[1]



  def getMean(self):
    mean_features = np.zeros((self.batch_len, self.num_feat))
    for i in range(self.batch_len):
      mean_features_temp = np.mean(self.batch_list[i], axis=0)
      mean_features[i]  = mean_features_temp
    print("mean_features shape: ", mean_features.shape)
    return mean_features

  def getStd(self):
    std_features = np.zeros((self.batch_len, self.num_feat))
    for i in range(self.batch_len):
      std_features_temp = np.std(self.batch_list[i], axis=0)
      std_features[i]  = std_features_temp
    print("std_features shape: ", std_features.shape)
    return std_features

  def getNormalizedClassBatches(self):
    normalized = np.zeros((self.batch_len, self.num_feat))
    mean_features = self.getMean() 
    std_features = self.getStd()
    for i range(self.batch_len):
      normtemp
      norm=(self.batch_list[0]-mean_features[0])/std_features[0]
    return norm

    
    #print("feature_score shape: ", feature_score.shape)
    #norm_examples =  
    #batch = ClassifiedBatches()
    return
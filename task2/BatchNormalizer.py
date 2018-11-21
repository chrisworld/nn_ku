import numpy as np
from ClassifiedBatches import ClassifiedBatches

class BatchNormalizer():
  def __init__(self, examples, classes=None, batch_size=40, shuffle=False):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.batch_num = np.round(self.examples.shape[0]/batch_size)  #number of batches corresponding to batch_size
    self.shuffle = shuffle




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
    batch_list = np.array_split(self.norm,self.batch_num)
    print("single batch shape: ", batch_list[0].shape)
    return batch_list


    
    #print("feature_score shape: ", feature_score.shape)
    #norm_examples =  
    #batch = ClassifiedBatches()
    return
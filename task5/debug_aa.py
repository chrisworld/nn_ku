from ErrorCollector import ErrorCollector
from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from ResNetModel import ResNetModel
from RnnModel import RnnModel
from RnnModelTester import RnnModelTester

from nn18_ex2_load import load_isolet

import numpy as np
import logging
import os


# Main function
if __name__ == '__main__':

  X, C, X_tst, C_tst = load_isolet()

  # setup logging 
  log_file_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'logs' + os.sep
  log_file_name = "Model.log"
  if not os.path.exists(log_file_path):
      os.makedirs(log_file_path)
  logging.basicConfig(filename=log_file_path + log_file_name, level=logging.INFO)








  # Parameters and model
  epochs = 100
  learning_rate = [1e-3]
  n_hidden = 40 # number of hidden units within layer
  n_layer = 1   # number of hidden layers
  activation = ['relu', 'tanh']
  batch_size = 20
  max_sequence_length = 1000

  # Batch Normalizer
  #bn = BatchNormalizer(X, C, batch_size=batch_size, shuffle=True)
  #train_batches = bn.getBatches(X, C)
  #test_batches = bn.getBatches(X_tst, C_tst, test=True)
  # use Test set as Validation
  #train_batches.examples_validation = test_batches.examples_validation
  #train_batches.classes_validation = test_batches.classes_validation

  model_tester = RnnModelTester(epochs, learning_rate, max_sequence_length)
  model_tester.run(train_batches, test_batches)




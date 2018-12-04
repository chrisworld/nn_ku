from ErrorCollector import ErrorCollector
from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model
from ModelTester import ModelTester

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
  epochs = 80
  learning_rates = [9e-5]#,11e-5,13e-5,15e-5]
  n_hidden = [40] # number of hidden units within layer
  n_layer = [9]   # number of hidden layers
  activation = ['relu', 'tanh']
  batch_size = 20

  # Batch Normalizer
  bn = BatchNormalizer(X, C, batch_size=batch_size, shuffle=True)
  train_batches = bn.getBatches(X, C)
  test_batches = bn.getBatches(X_tst, C_tst, test=True)
  # use Test set as Validation
  train_batches.examples_validation = test_batches.examples_validation
  train_batches.classes_validation = test_batches.classes_validation

  model_tester = ModelTester(epochs, learning_rates, n_hidden, n_layer, activation=activation)
  model_tester.run(train_batches, test_batches)

  # run best model at best epoch with whole training set and use test set as validation
  #epochs = 10
  #learning_rates = [0.01]
  #n_hidden = [150]
  #n_layer = [1]
  #train_batches = bn.getBatches(X, C, is_validation=False)
  #model_tester = ModelTester(epochs, learning_rates, n_hidden, n_layer)
  #model_tester.run(train_batches, test_batches)


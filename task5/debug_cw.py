from ErrorCollector import ErrorCollector
from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model
from ResNetModel import ResNetModel
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
  epochs = 200
  learning_rate = [1e-5, 3e-5, 6e-5]
  n_hidden = [40] # number of hidden units within layer
  n_layer = [9]   # number of hidden layers
  #activation = ['relu', 'tanh']
  activation = ['relu']
  batch_size = 20

  # Batch Normalizer
  bn = BatchNormalizer(X, C, batch_size=batch_size, shuffle=True)
  train_batches = bn.getBatches(X, C)
  test_batches = bn.getBatches(X_tst, C_tst, test=True)
  # use Test set as Validation
  train_batches.examples_validation = test_batches.examples_validation
  train_batches.classes_validation = test_batches.classes_validation

  #model_tester = ModelTester(epochs, learning_rate, n_hidden, n_layer, activation=activation, is_res_net = False)
  #model_tester.run(train_batches, test_batches)

  resnet_tester = ModelTester(epochs, learning_rate, n_hidden, n_layer, activation=activation, is_res_net = True)
  # get the best from the other model tester
  #resnet_tester.best_test_acc = model_tester.best_test_acc
  #resnet_tester.best_model_param = model_tester.best_model_param
  resnet_tester.run(train_batches, test_batches)


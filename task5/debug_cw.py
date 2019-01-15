from ErrorCollector import ErrorCollector
from ReberBatches import ReberBatches
from Trainer import Trainer
from Evaluator import Evaluator
from RnnModelTester import RnnModelTester

from grammar import *

import numpy as np
import logging
import os


# Main function
if __name__ == '__main__':

  # setup logging 
  log_file_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'logs' + os.sep
  log_file_name = "Model.log"
  if not os.path.exists(log_file_path):
      os.makedirs(log_file_path)
  logging.basicConfig(filename=log_file_path + log_file_name, level=logging.INFO)

  # Parameters and model
  epochs = 200
  learning_rate = [1e-5]
  n_hidden = [14] # number of hidden units within layer
  n_layer = [1]   # number of hidden layers
  rnn_unit = ['simple', 'lstm', 'gru']

  # Get Grammar data
  batch_size = 2
  #n_train_samples = 5000
  #n_val_samples = 500
  #n_test_samples = 500

  n_train_samples = 10
  n_val_samples = 2
  n_test_samples = 2

  reber_batches = ReberBatches(n_train_samples, n_val_samples, n_test_samples, batch_size)
  #print (reber_batches.batch_examples_train)

  model_tester = RnnModelTester(epochs, learning_rate, n_layer, n_hidden)
  model_tester.run(reber_batches)
  #train_batches = bn.getBatches(X, C)
  #test_batches = bn.getBatches(X_tst, C_tst, test=True)

  # use Test set as Validation
  #train_batches.examples_validation = test_batches.examples_validation
  #train_batches.classes_validation = test_batches.classes_validation


  #rnn_tester = ModelTester(epochs, learning_rate, n_hidden, n_layer, activation=activation, is_res_net = True)


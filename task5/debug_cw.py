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
  epochs = 100
  learning_rates = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2]
  n_hidden = [14] # number of hidden units within layer
  n_layer = [1]   # number of hidden layers
  rnn_unit = ['lstm']
  adam_optimizer = True

  # Get Grammar data
  batch_size = 40
  n_train_samples = 5000
  n_val_samples = 500
  n_test_samples = 500

  # Create Reber Batches
  reber_batches = ReberBatches(n_train_samples, n_val_samples, n_test_samples, batch_size)

  # Model Testing
  model_tester = RnnModelTester(epochs, learning_rates, n_layer, n_hidden, rnn_unit=rnn_unit, adam_optimizer=adam_optimizer)
  model_tester.run(reber_batches)



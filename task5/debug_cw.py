from ErrorCollector import ErrorCollector
from ClassifiedBatches import ClassifiedBatches
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
  learning_rate = [1e-5, 3e-5, 6e-5]
  n_hidden = [40] # number of hidden units within layer
  n_layer = [1]   # number of hidden layers
  rnn_unit = ['simple', 'lstm', 'gru']
  batch_size = 20

  # Get Grammar data
  n_training_samples = 5000
  n_validation_samples = 500
  n_test_samples = 500

  for n in range(5):
    reber = make_embedded_reber()
    reber_one_hot = str_to_vec(reber)
    reber_next = str_to_next_embed(reber)
    print('\n\nReber String: ' + reber + 
      '\none hot: \n' + str(reber_one_hot) + 
      '\nnext : \n' + str(reber_next))
 
  #train_batches = bn.getBatches(X, C)
  #test_batches = bn.getBatches(X_tst, C_tst, test=True)

  # use Test set as Validation
  #train_batches.examples_validation = test_batches.examples_validation
  #train_batches.classes_validation = test_batches.classes_validation


  #rnn_tester = ModelTester(epochs, learning_rate, n_hidden, n_layer, activation=activation, is_res_net = True)


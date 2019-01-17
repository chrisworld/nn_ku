from ErrorCollector import ErrorCollector

from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from ReberBatches import ReberBatches
from RnnModel import RnnModel
from RnnModelTester import RnnModelTester



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
  epochs = 50
  #learning_rates = [1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,1e-2]
  learning_rates = [1e-1 ,1e-2, 1e-3, 1e-4, 1e-5]
  n_hidden = [14] # number of hidden units within layer
  n_layer = [1]   # number of hidden layers
  activation = ['relu', 'tanh']
  batch_size = 40

  n_train_samples, n_val_samples, n_test_samples = 5000, 500, 500

  reber_batches = ReberBatches(n_train_samples, n_val_samples, n_test_samples, batch_size)
  max_sequence_length = reber_batches.train_max_seq_len

  #model_tester = RnnModelTester(epochs, learning_rates, max_sequence_length, n_layer, n_hidden, n_symbols=7,
  #                              n_out=7, rnn_unit='lstm')
  model_tester = RnnModelTester(epochs, learning_rates, n_layer, n_hidden,adam_optimizer=False)
  model_tester.run(reber_batches)




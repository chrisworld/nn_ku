from ErrorCollector import ErrorCollector
from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model

from nn18_ex2_load import load_isolet

import numpy as np
import logging
import os


if __name__ == '__main__':

  # training and validation error collector
  ec = ErrorCollector()

  X, C, X_tst, C_tst = load_isolet()
  #print("X shape: ", X.shape[0])
  #print("X shape: ", X.shape[1])

  # Parameters and model
  epochs = 5
  #learning_rate = 0.001
  learning_rate = 0.01
  model = Model(n_in=X.shape[1], n_hidden=300, n_out=26, n_layer=1)
  batch_size = 40

  # setup logging 
  log_file_name = 'logs' + os.sep + 'Log' + '_ep' + str(epochs) + '_hidu' + str(model.n_hidden) + '_hidl' + str(model.n_layer) + '_lr' + str(learning_rate) + '.log'
  logging.basicConfig(filename=log_file_name, level=logging.INFO)

  # Batch Normalize
  bn = BatchNormalizer(X, C, batch_size=batch_size, shuffle=True)
  train_batches = bn.getBatches(X, C, is_validation=False)
  test_batches = bn.getBatches(X_tst, C_tst, test=True)

  print('examples to train: ', train_batches.examples_train.shape)

  # Training
  trainer = Trainer(model, train_batches, ec)
  trainer.train(learning_rate, epochs, early_stop_lim=25)

  # print error plots
  ec.plotTrainTestError(model, batch_size, learning_rate, epochs)
  ec.plotTrainTestAcc(model, batch_size, learning_rate, epochs)

  # Testing
  evaluator = Evaluator(model, test_batches, trainer.getSaveFilePath())
  evaluator.eval()


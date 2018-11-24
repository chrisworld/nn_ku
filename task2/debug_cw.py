from ErrorCollector import ErrorCollector
from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model

from nn18_ex2_load import load_isolet

import tensorflow as tf
import numpy as np
import logging


if __name__ == '__main__':

  # training and validation error collector
  ec = ErrorCollector()

  # BatchNormalizer
  X, C, X_tst, C_tst = load_isolet()
  print("X shape: ", X.shape)
  print("C shape: ", C.shape)
  print("X shape: ", X.shape[0])
  print("X shape: ", X.shape[1])

  # Parameters and model
  epochs = 6
  learning_rate = 0.001
  learning_rate = 0.5
  model = Model(n_in=X.shape[1], n_hidden=100, n_out=26, n_layer=0)
  batch_size = 40

  # setup logging 
  log_file_name = 'logs/' + 'Log' + '_ep' + str(epochs) + '_hidu' + str(model.n_hidden) + '_hidl' + str(model.n_layer) + '_lr' + str(learning_rate) + '.log'
  logging.basicConfig(filename=log_file_name, level=logging.INFO)

  # Batch Normalize
  bn = BatchNormalizer(X, C, batch_size=batch_size, shuffle=False)
  bn.getMean()
  bn.getStd()
  bn.getNormalized()
  train_batches = bn.getBatches()
  test_batches = bn.getBatches()

  # Training
  trainer = Trainer(model, train_batches, ec)
  trainer.train(learning_rate, epochs)

  # print error plots
  ec.plotTrainTestError(model, batch_size, learning_rate, epochs)
  ec.plotTrainTestAcc(model, batch_size, learning_rate, epochs)

  # Testing
  evaluator = Evaluator(model, test_batches)

  evaluator.eval()


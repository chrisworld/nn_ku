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


if __name__ == '__main__':

  X, C, X_tst, C_tst = load_isolet()

  # Parameters and model
  epochs = 50
  #learning_rates = [0.001, 0.01, 0.1]
  learning_rates = [0.01, 0.1]
  #n_hidden = [150, 300, 600]
  n_hidden = [150, 300]
  n_layer = [0, 1, 2]

  batch_size = 40

  # Batch Normalize
  bn = BatchNormalizer(X, C, batch_size=batch_size, shuffle=True)
  train_batches = bn.getBatches(X, C)
  test_batches = bn.getBatches(X_tst, C_tst, test=True)

  model_tester = ModelTester(epochs, learning_rates, n_hidden, n_layer)
  model_tester.run(train_batches, test_batches)


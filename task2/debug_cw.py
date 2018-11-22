from ErrorCollector import ErrorCollector
from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model
from nn18_ex2_load import load_isolet
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
  # test error collector
  ec = ErrorCollector()

  # BatchNormalizer
  X, C, X_tst, C_tst = load_isolet()
  print("X shape: ", X.shape)
  print("C shape: ", C.shape)
  print("X shape: ", X.shape[0])
  print("X shape: ", X.shape[1])
  # one hot

  learning_rate = 0.001
  model = Model(n_in=X.shape[1], n_hidden=100, n_out=26, n_layer=1)

  bn = BatchNormalizer(X, C, batch_size=40, shuffle=False)
  #print("mean: ", bn.getMean())
  #print("getStd: ", bn.getStd())
  bn.getMean()
  bn.getStd()
  bn.getNormalized()
  train_batches = bn.getBatches()

  #X_list = np.array_split(X, 2)
  #C_list = np.array_split(C, 2)

  print("X_list_pre: ", train_batches.examples[0].shape)
  #print("C_list_pre: ", C_list[0].shape)

  #batches = ClassifiedBatches(X[:], C[:], X.shape[0])
  #test_batches = ClassifiedBatches(X_tst, C_tst, X.shape[0])

  #print("X_list: ", batches.examples[0].shape)
  #print("C_list: ", batches.classes_one_hot[0].shape)


  trainer = Trainer(model, train_batches, ec)
  evaluator = trainer.train(learning_rate, epochs=2)
  #evaluator.batches = training_batches
  evaluator.eval()


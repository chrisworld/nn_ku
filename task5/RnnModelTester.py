from ErrorCollector import ErrorCollector
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from RnnModel import RnnModel

import logging
import os

# Model Tester class
class RnnModelTester():
  def __init__(self, epochs, learning_rates, n_layer, n_hidden, n_symbols=7,
               n_out=7, rnn_unit='lstm'):
    self.epochs = epochs
    self.learning_rates = learning_rates
    self.n_hidden = n_hidden
    self.n_layer = n_layer
    self.models = []
    self.n_out = n_out
    self.best_test_acc = 0
    self.best_model_param = "None"
    self.n_symbols = n_symbols
    self.rnn_unit = rnn_unit

  def run(self, batches):
    # training and validation error collector
    ec = ErrorCollector()

    print("-----ModelTester-----")
    logging.info("-----ModelTester-----")



    for n_hidden in self.n_hidden:
      for n_layer in self.n_layer:
        print("RNN with max seq len: ", batches.max_seq_len)
        self.models.append(RnnModel(self.n_symbols, n_hidden, self.n_out,
                                    self.rnn_unit, batches.max_seq_len, n_layer))
    
    for model in self.models:
      trainer = Trainer(model, batches, ec)
      for learning_rate in self.learning_rates:
        trainer.train(learning_rate, self.epochs, early_stop_lim=25)
        # print error plots
        ec.plotTrainTestError(model, batches.batch_size, learning_rate, self.epochs)
        ec.plotTrainTestAcc(model, batches.batch_size, learning_rate, self.epochs)
        ec.resetErrors()
        evaluator = Evaluator(model, batches, trainer.getSaveFilePath())
        test_loss, test_acc  = evaluator.eval()
        trainer.resetBestScore()

        if self.best_test_acc == 0 or test_acc > self.best_test_acc:
          self.best_test_acc = test_acc
          self.best_model_param = 'Param_' + model.name + '_ep-' + str(self.epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate)

    print("-----ModelTester finished, best test acc: [%.6f] with model: %s " % (self.best_test_acc, self.best_model_param))
    logging.info("-----ModelTester finished, best test acc: [%.6f] with model: %s " % (self.best_test_acc, self.best_model_param))


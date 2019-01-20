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
               n_out=7, rnn_unit='lstm', adam_optimizer=True):
    self.epochs = epochs
    self.learning_rates = learning_rates
    self.n_hidden = n_hidden
    self.n_layer = n_layer
    self.models = []
    self.n_out = n_out
    self.best_test_loss = 0
    self.best_conv_time = 100000
    self.best_model_param = "None"
    self.n_symbols = n_symbols
    self.rnn_unit = rnn_unit
    self.adam_optimizer = adam_optimizer

  def run(self, batches):
    # training and validation error collector
    ec = ErrorCollector()

    print("-----ModelTester-----")
    logging.info("-----ModelTester-----")

    # create models
    for n_hidden in self.n_hidden:
      for n_layer in self.n_layer:
        for rnn_unit in self.rnn_unit:
            print("RNN with max seq len: ", batches.max_seq_len)
            self.models.append(RnnModel(self.n_symbols, n_hidden, self.n_out,
                                        rnn_unit, batches.max_seq_len, n_layer, self.adam_optimizer))
    
    # training and evaluation
    plot_id = 0
    for model in self.models:
      trainer = Trainer(model, batches, ec)
      for learning_rate in self.learning_rates:
        plot_id += 1
        trainer.train(learning_rate, self.epochs, adam_optimizer=self.adam_optimizer, early_stopping=True, early_stop_lim=10)
        # print error plots
        ec.plotTrainTestError(model, batches.batch_size, learning_rate, self.epochs, plot_id)
        ec.resetErrors()
        evaluator = Evaluator(model, batches, trainer.getSaveFilePath())
        test_loss = evaluator.eval()

        # get best conversion time
        if trainer.best_epoch < self.best_conv_time:
          self.best_conv_time = trainer.best_epoch
          self.best_model_param = 'Param_' + model.name + '_ep-' + str(self.epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '_id-' + str(plot_id)

        # reset scores
        trainer.resetBestScore()

    # print convergence times
    ec.convergenceTimeMeanAndStd()

    print("-----ModelTester finished, best test error: [%.6f] and conv time (epochs): [%i] with model: %s " % (self.best_test_loss, self.best_conv_time, self.best_model_param))
    logging.info("-----ModelTester finished, best test error: [%.6f] and conv time (epochs): [%i] with model: %s " % (self.best_test_loss, self.best_conv_time, self.best_model_param))


from ErrorCollector import ErrorCollector
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from RnnModel import RnnModel

import logging
import os

# Model Tester class
class RnnModelTester():
  def __init__(self, epochs, learning_rates, n_hidden, n_layer, n_in=300, n_out=26, activation='relu', is_res_net = False):
    self.epochs = epochs
    self.learning_rates = learning_rates
    self.n_hidden = n_hidden
    self.n_layer = n_layer
    self.models = []
    self.n_in = n_in
    self.n_out = n_out
    self.best_test_acc = 0
    self.best_model_param = "None"
    self.activation = activation
    self.is_res_net = is_res_net

  def run(self, train_batches, test_batches):
    # training and validation error collector
    ec = ErrorCollector()

    print("-----ModelTester-----")
    logging.info("-----ModelTester-----")

    for n_hidden in self.n_hidden:
      for n_layer in self.n_layer:
        if self.is_res_net:
          self.models.append(ResNetModel(n_in=self.n_in, n_hidden=n_hidden, n_out=self.n_out, n_layer=n_layer, activation='relu'))
        else:
          for activation in self.activation:
            # create models
            self.models.append(Model(n_in=self.n_in, n_hidden=n_hidden, n_out=self.n_out, n_layer=n_layer, activation=activation))
          
    
    for model in self.models:
      trainer = Trainer(model, train_batches, ec)
      for learning_rate in self.learning_rates:
        trainer.train(learning_rate, self.epochs, early_stop_lim=25)
        # print error plots
        ec.plotTrainTestError(model, train_batches.batch_size, learning_rate, self.epochs, model.activation)
        ec.plotTrainTestAcc(model, train_batches.batch_size, learning_rate, self.epochs, model.activation)
        ec.resetErrors()
        evaluator = Evaluator(model, test_batches, trainer.getSaveFilePath())
        test_loss, test_acc  = evaluator.eval()
        trainer.resetBestScore()

        if self.best_test_acc == 0 or test_acc > self.best_test_acc:
          self.best_test_acc = test_acc
          self.best_model_param = 'Param_' + model.name + '_ep-' + str(self.epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate)

    print("-----ModelTester finished, best test acc: [%.6f] with model: %s " % (self.best_test_acc, self.best_model_param))
    logging.info("-----ModelTester finished, best test acc: [%.6f] with model: %s " % (self.best_test_acc, self.best_model_param))


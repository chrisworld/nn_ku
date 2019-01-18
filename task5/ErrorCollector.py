import matplotlib.pyplot as plt
import os
import numpy as np
import logging

class ErrorCollector():
  def __init__(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []
    self.convergence_time_list = []
  
  def resetErrors(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []

  def addConvergenceTime(self, epoch):
    #test_error_array = np.array(self.test_error_list)
    #epoch_of_convergence = np.nonzero(test_error_array<1e-7)[0][0] + 1
    #self.convergence_time_list.append(epoch_of_convergence)
    self.convergence_time_list.append(epoch)

  def convergenceTimeMeanAndStd(self):
    convergence_time_mean = np.mean(self.convergence_time_list)
    convergence_time_std = np.std(self.convergence_time_list)
    print("-----ModelTester Convergence Time of Models-----")
    print("convergence time in epochs\nmean:                 [%.2f]\n standard deviation:  [%.2f]" %(convergence_time_mean,convergence_time_std))
    logging.info("convergence time in epochs\nmean:                 [%.2f]\n standard deviation:  [%.2f]" %(convergence_time_mean,convergence_time_std))

    return convergence_time_mean, convergence_time_std



  def addTrainError(self, train_error):
    self.train_error_list.append(train_error)

  def addTestError(self, test_error):   
    self.test_error_list.append(test_error)

  def addTrainAcc(self, train_acc):
    self.train_acc_list.append(1-train_acc) # actually it's misclassification

  def addTestAcc(self, test_acc):
    self.test_acc_list.append(1-test_acc)

  def plotTrainTestError(self, model, batch_size, learning_rate, epochs, plot_id):
    print("Plot Errors")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_error_list, color='blue', label='training', lw=2)
    ax.plot(self.test_error_list, color='red', label='test/validation', lw=1.7, linestyle= '--',alpha=0.9)
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy loss')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'plots' + os.sep
    save_name = 'Loss_' + model.name + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_opt-'+ str(model.optimizer_name) + '_lr-' + str(learning_rate) + '_id-' + str(plot_id) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()

  def plotTrainTestAcc(self, model, batch_size, learning_rate, epochs):
    print("Plot Misclassification")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_acc_list, color='blue', label='training', lw=2)
    ax.plot(self.test_acc_list, color='green', label='test', lw=2)

    ax.set_autoscaley_on(False)
    ax.set_ylim([0, 1])
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Misclassification')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'plots' + os.sep
    save_name = 'Misclass_'+ model.name + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()

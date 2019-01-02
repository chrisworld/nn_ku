import matplotlib.pyplot as plt
import os

class ErrorCollector():
  def __init__(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []
  
  def resetErrors(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []

  def addTrainError(self, train_error):
    self.train_error_list.append(train_error)

  def addTestError(self, test_error):   
    self.test_error_list.append(test_error)

  def addTrainAcc(self, train_acc):
    self.train_acc_list.append(1-train_acc) # actually it's misclassification

  def addTestAcc(self, test_acc):
    self.test_acc_list.append(1-test_acc)

  def plotTrainTestError(self, model, batch_size, learning_rate, epochs, activation='relu'):
    print("Plot Errors")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_error_list, color='blue', label='training', lw=2)
    ax.plot(self.test_error_list, color='green', label='test', lw=2)
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy loss')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'plots' + os.sep
    save_name = 'Loss_' + model.name + '_act-'+ activation + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()

  def plotTrainTestAcc(self, model, batch_size, learning_rate, epochs,activation='relu'):
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
    save_name = 'Misclass_'+ model.name + '_act-'+ activation + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()

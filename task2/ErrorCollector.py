import matplotlib.pyplot as plt

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
    self.train_acc_list.append(train_acc)

  def addTestAcc(self, test_acc):
    self.test_acc_list.append(test_acc)

  def plotTrainTestError(self, model, batch_size, learning_rate, epochs):
    print("Plot Errors")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_error_list, color='blue', label='training', lw=2)
    ax.plot(self.test_error_list, color='green', label='validation', lw=2)
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy loss')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_name = 'plots/' + 'Loss' + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    #plt.show()

  def plotTrainTestAcc(self, model, batch_size, learning_rate, epochs):
    print("Plot Accuracy")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_acc_list, color='blue', label='training', lw=2)
    ax.plot(self.test_acc_list, color='green', label='validation', lw=2)

    ax.set_autoscaley_on(False)
    ax.set_ylim([0, 1])
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Accuracy')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_name = 'plots/' + 'Accuracy' + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    #plt.show()

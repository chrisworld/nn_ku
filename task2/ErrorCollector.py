import matplotlib.pyplot as plt

class ErrorCollector():
  def __init__(self):
    self.train_error_list = []
    self.test_error_list = []
  
  def resetErrors(self):
    self.train_error_list = []
    self.test_error_list = []

  def addTrainError(self, train_error):
    self.train_error_list.append(train_error)

  def addTestError(self, test_error):   
    self.test_error_list.append(test_error)

  def plotTrainTestError(self, model, batch_size, learning_rate, epochs):
    print("Plot errors")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_error_list, color='blue', label='training', lw=2)
    ax.plot(self.test_error_list, color='green', label='testing', lw=2)
    ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy Loss')
    plt.legend()
    #TODO: name
    plt.savefig('plots/foo.png', dpi=150, bbox_inches='tight')
    #plt.show()

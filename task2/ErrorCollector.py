import matplotlib.pyplot as plt

class ErrorCollector():
  def __init__(self):
    self.train_error_list = []
    self.test_error_list = []
  
  def addTrainError(self, train_error):
    self.train_error_list.append(train_error)

  def addTestError(self, test_error):   
    self.test_error_list.append(test_error)

  def plotTrainTestError(self):
    print("Plot errors")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_error_list, color='blue', label='training', lw=2)
    ax.plot(self.test_error_list, color='green', label='testing', lw=2)
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy')
    plt.legend()
    plt.show()

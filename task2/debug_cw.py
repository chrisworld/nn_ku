from ErrorCollector import ErrorCollector

if __name__ == '__main__':
  # test error collector
  ec = ErrorCollector()
  for x in range(0, 10):
    #print(x)
    ec.addTrainError(x)
    ec.addTestError(10-x)
  ec.plotTrainTestError()
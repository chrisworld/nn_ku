from ErrorCollector import ErrorCollector
from BatchNormalizer import BatchNormalizer
from nn18_ex2_load import load_isolet

if __name__ == '__main__':
  # test error collector
  ec = ErrorCollector()
  for x in range(0, 10):
    #print(x)
    ec.addTrainError(x)
    ec.addTestError(10-x)
  #ec.plotTrainTestError()

  # BatchNormalizer
  X, C, X_tst, C_tst = load_isolet()

  print("X shape: ", X.shape)
  print("C shape: ", C.shape)

  bn = BatchNormalizer(X, C, batch_size=40, shuffle=False)
  bn.getNormalizedClassBatches()

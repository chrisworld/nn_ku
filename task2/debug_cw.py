from ErrorCollector import ErrorCollector
#from BatchNormalizer import BatchNormalizer
from ClassifiedBatches import ClassifiedBatches
from Trainer import Trainer
from Evaluator import Evaluator
from Model import Model
from nn18_ex2_load import load_isolet
import tensorflow as tf

if __name__ == '__main__':
  # test error collector
  ec = ErrorCollector()

  # BatchNormalizer
  X, C, X_tst, C_tst = load_isolet()
  print("X shape: ", X.shape)
  print("C shape: ", C.shape)
  print("X shape: ", X.shape[0])
  print("X shape: ", X.shape[1])
  # one hot
  classes_one_hot = tf.one_hot(C-1, 26)
  classes_tst_one_hot = tf.one_hot(C_tst-1, 26)
  sess = tf.Session()
  c_oh = sess.run(classes_one_hot)
  c_tst_oh = sess.run(classes_tst_one_hot)

  print("one hot class: ", c_oh.shape)
  print("one hot test class: ", c_tst_oh.shape)

  learning_rate = 0.001
  model = Model(n_in=X.shape[1], n_hidden=100, n_out=26, n_layer=1)

  #bn = BatchNormalizer(X, C, batch_size=40, shuffle=False)
  #bn.getNormalizedClassBatches()

  batches = ClassifiedBatches(X, c_oh, X.shape[0])
  test_batches = ClassifiedBatches(X_tst, c_tst_oh, X.shape[0])

  trainer = Trainer(model, batches, ec)
  trainer.train(learning_rate, epochs=2)

  evaluator = Evaluator(model, test_batches, ec)
  evaluator.eval()


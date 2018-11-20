# Trainer class for training Neural Networks
import tensorflow as tf
import logging

class Trainer():
  def __init__(self, model, batches, error_collector):
    self.model = model
    self.batches = batches
    self.error_collector = error_collector

  def train(self, learning_rate, epochs):
    # setup logging 
    logging.basicConfig(filename='logs/example.log',level=logging.INFO)
    logging.info('\n \nTrainer Logger \n' + 'Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))

    # setup training
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.model.cross_entropy)
    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # init variables
    init = tf.global_variables_initializer() 
    sess = tf.Session()
    sess.run(init)

    # run epochs
    for k in range(epochs):
      # training step
      sess.run(train_step, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      # Compute the errors over the whole dataset
      train_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      # Compute the acc over the whole dataset
      train_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      self.error_collector.addTrainError(train_loss)
      print("Iteration: ",k, " train loss: ",train_loss, "train acc: ", train_acc)
      logging.info("Iteration: " + str(k) + " train loss: " + str(train_loss) + " train acc: " + str(train_acc))

    # print errors
    self.error_collector.plotTrainTestError(self.model, self.batches.batch_size, learning_rate, epochs)
# Trainer class for training Neural Networks
import tensorflow as tf
import logging
import os


from Evaluator import Evaluator

class Trainer():
  def __init__(self, model, batches, error_collector):
    self.model = model
    self.batches = batches
    self.error_collector = error_collector
    self.best_loss = 0
    #self.save_path = os.path.realpath(__file__)
    self.save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'tmp' + os.sep

    #self.save_path = os.path.normpath(join(os.getcwd(), path)) + '/tmp' 

  def train(self, learning_rate, epochs):
    # logger info
    logging.info('\n \nTrainer Logger \n' + 'Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))

    # setup training
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.model.cross_entropy)
    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # init variables
    init = tf.global_variables_initializer() 
    # save variables
    saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(init)

      # run epochs
      for k in range(epochs):
        # gradient over each batch
        for batch_example, batch_class in zip(self.batches.batch_examples, self.batches.batch_classes):
          # training step
          sess.run(train_step, feed_dict={self.model.x: batch_example, self.model.z_: batch_class})
        
        # Compute the errors and acc over the training dataset
        train_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
        train_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
        self.error_collector.addTrainError(train_loss)
        self.error_collector.addTrainAcc(train_acc)
        # Compute the errors and acc of the validation set
        test_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
        test_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
        self.error_collector.addTestError(test_loss)
        self.error_collector.addTestAcc(test_acc)

        # Early stopping
        if self.best_loss == 0 or test_loss < self.best_loss:
          file_name = os.path.normpath('Param_ep-' + str(epochs) + '_hidu-' + str(self.model.n_hidden) + '_hidl-' + str(self.model.n_layer) + '_lr-' + str(learning_rate) + '.ckpt')
          print("Model saved: %s" % self.save_path + file_name)
          save_path = saver.save(sess, self.save_path + file_name)
          self.best_loss = test_loss

        # logging
        print("Iteration: ",k, " train loss: ",train_loss, "train acc: ", train_acc, " valid loss: ", test_loss, " valid acc: ", test_acc)
        logging.info("Iteration: " + str(k) + " train loss: " + str(train_loss) + " train acc: " + str(train_acc) + " val loss: " + str(test_loss) + " val acc: " + str(test_acc))
    
    # finish log
    print("-----Training finished with best validation loss: ", self.best_loss)
    logging.info("-----Training finished with best validation loss: " + str(self.best_loss))


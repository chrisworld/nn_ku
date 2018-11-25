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
    self.best_validation_loss = 0
    self.best_validation_acc = 0
    self.best_epoch = 0
    #self.save_path = os.path.realpath(__file__)
    self.save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'tmp' + os.sep
    self.file_name = ""

    #self.save_path = os.path.normpath(join(os.getcwd(), path)) + '/tmp' 

  def train(self, learning_rate, epochs, early_stop_lim=1000):
    # save parameter file name
    self.file_name = 'Param_ep-' + str(epochs) + '_hidu-' + str(self.model.n_hidden) + '_hidl-' + str(self.model.n_layer) + '_lr-' + str(learning_rate) + '.ckpt'
    
    # setup training
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.model.cross_entropy)
    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # init variables
    init = tf.global_variables_initializer() 
    # save variables
    saver = tf.train.Saver()

    # logging infos
    print("-----Training-----")
    print('Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))
    logging.info("-----Training-----")
    logging.info('Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))

    early_stop_counter = 0
    with tf.Session() as sess:
      sess.run(init)

      # run epochs
      for k in range(epochs):
        # gradient over each batch
        for batch_example, batch_class in zip(self.batches.batch_examples_train, self.batches.batch_classes_train):
          # training step
          sess.run(train_step, feed_dict={self.model.x: batch_example, self.model.z_: batch_class})
        
        # Compute the errors and acc over the training dataset
        train_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples_train, self.model.z_: self.batches.classes_train})
        train_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples_train, self.model.z_: self.batches.classes_train})
        self.error_collector.addTrainError(train_loss)
        self.error_collector.addTrainAcc(train_acc)
        # Compute the errors and acc of the validation set
        test_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples_validation, self.model.z_: self.batches.classes_validation})
        test_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples_validation, self.model.z_: self.batches.classes_validation})
        self.error_collector.addTestError(test_loss)
        self.error_collector.addTestAcc(test_acc)

        # Early stopping, save best parameters
        if self.batches.is_validation == True:
          if self.best_validation_loss == 0 or test_loss < self.best_validation_loss:
            #print("---Model saved: %s" % self.save_path + self.file_name)
            saver.save(sess, self.save_path + self.file_name)
            self.best_validation_loss = test_loss
            self.best_validation_acc = test_acc
            self.best_epoch = k
            early_stop_counter = 0
          else:
            early_stop_counter += 1

          # stop the training if no improvement
          if early_stop_counter > early_stop_lim: 
            print("---end due to early stopping limit")
            logging.info("---end due to early stopping limit")
            break

        # logging iterations
        print("Iteration: ",k, " train loss: [%.4f]" % train_loss, " train acc: [%.4f]" % train_acc, " valid loss: [%.4f]" % test_loss, " valid acc: [%.4f]" % test_acc)
        logging.info("Iteration: %i" % k + " train loss: [%.4f]" % train_loss + " train acc: [%.4f]" % train_acc + " valid loss: [%.4f]" % test_loss + " valid acc: [%.4f]" % test_acc)
      
      if self.batches.is_validation == False:
        saver.save(sess, self.save_path + self.file_name)


    # finish log
    print("-----Training finished with best validation loss: [%.4f] validation acc: [%.4f] at epoch:[%i]" %(self.best_validation_loss, self.best_validation_acc, self.best_epoch) )
    logging.info("-----Training finished with best validation loss: [%.4f] validation acc: [%.4f] at epoch:[%i]" %(self.best_validation_loss, self.best_validation_acc, self.best_epoch ) )

  # returns the path of saved model
  def getSaveFilePath(self):
    return self.save_path + self.file_name

  def resetBestScore(self):
    self.best_validation_loss = 0
    self.best_validation_acc = 0
    self.best_epoch = 0


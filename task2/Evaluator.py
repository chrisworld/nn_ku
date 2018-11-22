import tensorflow as tf
import logging

class Evaluator():
  def __init__(self, model, batches, error_collector):
    self.model = model
    self.batches = batches
    self.error_collector = error_collector
    # best loss and model, for early stopping
    self.best_loss = 0
    self.best_model = model

  def eval(self, it=None, is_validation=False):

    # evaluation
    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # init variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # compute scores on validation set
    test_loss = 0
    test_acc = 0
    if is_validation:
      test_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      test_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      self.error_collector.addTestError(test_loss)
      self.error_collector.addTestAcc(test_acc)
      # calculate best loss
      # TODO
      if self.best_loss == 0 or test_loss < self.best_loss:
        #saver = tf.train.Saver()
        #save_path = saver.save(sess, "./tmp/model.ckpt")
        #print("Model saved in path: %s" % save_path)

        #self.best_model = self.model
        self.best_loss = test_loss
        #print("TF Vars: ", tf.trainable_variables())
    
    # compute scores on test set 
    else:
      test_loss = sess.run(self.best_model.cross_entropy, feed_dict={self.best_model.x: self.batches.examples, self.best_model.z_: self.batches.classes})
      test_acc = sess.run(accuracy, feed_dict={self.best_model.x: self.batches.examples, self.best_model.z_: self.batches.classes})

    if is_validation:
      print("Iteration: ", it, " valid loss: ", test_loss, " valid acc: ", test_acc)
      logging.info("Iteration: " + str(it) + " val loss: " + str(test_loss) + " val acc: " + str(test_acc))
    else:
      print("test loss: ",test_loss, " test acc: ", test_acc)
      logging.info("test loss: " + str(test_loss) + "test acc: " + str(test_acc))
      #logging.info("best training loss: " + str(self.best_loss))



import tensorflow as tf
import logging

# Evaluator class
class Evaluator():
  def __init__(self, model, batches, save_path):
    self.model = model
    self.batches = batches
    self.save_path = save_path

  def eval(self):

    # evaluation
    correct_prediction = tf.equal(self.model.z_, tf.maximum(tf.sign(self.model.last_outputs), 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # restore parameters
    saver = tf.train.Saver()

    print("-----Evaluation of Test set-----")
    logging.info("-----Evaluation of Test set-----")
    
    # evaluate tf graph
    with tf.Session() as sess:
      saver.restore(sess, self.save_path)
      #test_loss = sess.run(self.model.error, feed_dict={self.model.X: self.batches.examples_test, self.model.z_: self.batches.targets_test})
      test_loss = sess.run(self.model.error, feed_dict={self.model.X: self.batches.examples_test,
                                                        self.model.z_: self.batches.targets_test,
                                                        self.model.seq_length: self.batches.seq_len_test})
      #test_acc = sess.run(accuracy, feed_dict={self.model.X: self.batches.examples_test, self.model.z_: self.batches.targets_test})
      print("test loss: [%.6f]" % test_loss)#, " test acc: [%.6f]" % test_acc)
      logging.info("test loss: [%.6f]" % test_loss)# + " test acc: [%.6f]" % test_acc)

    return test_loss#, test_acc


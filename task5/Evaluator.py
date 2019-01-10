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
    correct_prediction = tf.equal(self.model.z_, tf.maximum(tf.sign(self.model.z), 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # restore parameters
    saver = tf.train.Saver()

    print("-----Evaluation of Test set-----")
    logging.info("-----Evaluation of Test set-----")
    
    # evaluate tf graph
    with tf.Session() as sess:
      saver.restore(sess, self.save_path)
      test_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      test_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      print("test loss: [%.6f]" % test_loss, " test acc: [%.6f]" % test_acc)
      logging.info("test loss: [%.6f]" % test_loss + " test acc: [%.6f]" % test_acc)

    return test_loss, test_acc


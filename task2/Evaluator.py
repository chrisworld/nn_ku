import tensorflow as tf
import logging

class Evaluator():
  def __init__(self, model, batches):
    self.model = model
    self.batches = batches

  def eval(self):

    # evaluation
    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # restore parameters
    init = tf.global_variables_initializer() 

    # evaluate tf graph
    with tf.Session() as sess:

      sess.run(init)

      test_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      test_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
      print("test loss: ",test_loss, " test acc: ", test_acc)
      logging.info("test loss: " + str(test_loss) + "test acc: " + str(test_acc))



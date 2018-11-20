import tensorflow as tf

class Evaluator():
  def __init__(self, model, batches, error_collector):
    self.model = model
    self.batches = batches
    self.error_collector = error_collector
    self.best_score = 0

  def eval(self):
 
    # setup training
    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # init variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # compute scores
    test_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
    test_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples, self.model.z_: self.batches.classes})
    self.error_collector.addTestError(test_loss)
    print("test loss: ",test_loss, " test acc: ", test_acc)
    #logging.info("Iteration: " + str(k) + " train loss: " + str(train_loss) + "train acc: " + str(train_acc))

  # TODO: saveParameters

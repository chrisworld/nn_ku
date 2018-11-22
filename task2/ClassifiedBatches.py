import tensorflow as tf

class ClassifiedBatches():
  def __init__(self, examples, classes, batch_size, num_classes=26):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size

    c_oh = tf.one_hot(classes-1, num_classes)
    with tf.Session() as sess:
      self.classes_one_hot = sess.run(c_oh)



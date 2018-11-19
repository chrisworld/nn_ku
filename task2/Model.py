import tensorflow as tf
import numpy as np
import numpy.random as rd

class Model():
  def __init__(self, n_in, n_hidden, n_out):
    self.n_in = n_in
    self.n_hidden = n_hidden
    self.n_out = n_out

    # Set the variables
    self.W_hid = tf.Variable(rd.randn(self.n_in, self.n_hidden) / np.sqrt(self.n_in),trainable=True)
    self.b_hid = tf.Variable(np.zeros(self.n_hidden),trainable=True)
    self.w_out = tf.Variable(rd.randn(self.n_hidden, self.n_out) / np.sqrt(self.n_in),trainable=True)
    self.b_out = tf.Variable(np.zeros(self.n_out))

    # Define the neuron operations
    self.x = tf.placeholder(shape=(None, self.n_in),dtype=tf.float64)
    self.y = tf.nn.tanh(tf.matmul(self.x, self.W_hid) + self.b_hid)
    self.z = tf.nn.softmax(tf.matmul(self.y, self.w_out) + self.b_out)

    self.z_ = tf.placeholder(shape=(None, self.n_out),dtype=tf.float64)
    self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.z_ * tf.log(self.z), reduction_indices=[1]))



  

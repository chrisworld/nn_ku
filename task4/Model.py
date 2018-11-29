import tensorflow as tf
import numpy as np
import numpy.random as rd

class Model():
  def __init__(self, n_in, n_hidden, n_out, n_layer=1):
    self.n_in = n_in
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.n_layer = n_layer

    # Weights, biases and activations
    self.W = []
    self.b = []
    self.h = []

    print('---Create Model--- ')
    # create list of weights and biases upon each layers
    for layer in range(n_layer + 1):
      # set connections upon layer
      if layer == 0: 
        print('Input W/b in layer: ', layer)
        if n_layer == 0:
          self.W.append(tf.Variable(rd.randn(self.n_in, self.n_out) / np.sqrt(self.n_in), trainable=True))
          self.b.append(tf.Variable(np.zeros(self.n_out), trainable=True))
        else:
          self.W.append(tf.Variable(rd.randn(self.n_in, self.n_hidden) / np.sqrt(self.n_in), trainable=True))
          self.b.append(tf.Variable(np.zeros(self.n_hidden), trainable=True))
      elif layer == n_layer:
        print('Output W/b in layer: ', layer)
        self.W.append(tf.Variable(rd.randn(self.n_hidden, self.n_out) / np.sqrt(self.n_hidden), trainable=True))
        self.b.append(tf.Variable(np.zeros(self.n_out), trainable=True))
      else:
        print('Hidden W/b in layer: ', layer)
        self.W.append(tf.Variable(rd.randn(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden), trainable=True))
        self.b.append(tf.Variable(np.zeros(self.n_hidden), trainable=True))

    # Define the neuron operations
    # input layer
    self.x = tf.placeholder(shape=(None, self.n_in),dtype=tf.float64)
    # connections and activation of hidden layer
    for layer in range(n_layer):
      if layer == 0:
        print('Activation h of layer: ', layer)
        self.h.append(tf.nn.tanh(tf.matmul(self.x, self.W[0]) + self.b[0]))
      else:
        print('Activation h of layer: ', layer)
        self.h.append(tf.nn.tanh(tf.matmul(self.h[layer-1], self.W[layer]) + self.b[layer]))

    # output activation
    if n_layer == 0:
      self.z = tf.nn.softmax(tf.matmul(self.x, self.W[n_layer]) + self.b[n_layer])
    else:
      self.z = tf.nn.softmax(tf.matmul(self.h[n_layer-1], self.W[n_layer]) + self.b[n_layer])

    # labels
    self.z_ = tf.placeholder(shape=(None, self.n_out),dtype=tf.float64)

    #loss
    self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.z_ * tf.log(self.z), reduction_indices=[1]))

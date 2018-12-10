import tensorflow as tf
import numpy as np
import numpy.random as rd

class ResNetModel():
  def __init__(self, n_in, n_hidden, n_out, n_layer=1, activation='relu'):
    self.name = 'ResNet_Model'
    self.n_in = n_in
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.n_layer = n_layer 
    self.n_resnet_blocks = int(n_layer/2)
    self.activation = activation

    # Weights, biases and activations
    self.W = []
    self.b = []
    self.h = []

    print('---Create ResNet Model--- ')
    # create list of weights and biases upon each layers
    # input layer
    print('Input W/b')
    self.W.append(tf.Variable(rd.randn(self.n_in, self.n_hidden) / np.sqrt(self.n_in), trainable=True))
    self.b.append(tf.Variable(np.zeros(self.n_hidden), trainable=True))

    if self.n_resnet_blocks != 0:
      # create resnet blocks
      for block in range(self.n_resnet_blocks):
        print('ResNet W/b  at Block: ', block)
        self.W.append(tf.Variable(rd.randn(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden), trainable=True))
        self.b.append(tf.Variable(np.zeros(self.n_hidden), trainable=True))
        self.W.append(tf.Variable(rd.randn(self.n_hidden, self.n_hidden) / np.sqrt(self.n_hidden), trainable=True))
        self.b.append(tf.Variable(np.zeros(self.n_hidden), trainable=True))

    print('Output W/b')
    self.W.append(tf.Variable(rd.randn(self.n_hidden, self.n_out) / np.sqrt(self.n_hidden), trainable=True))
    self.b.append(tf.Variable(np.zeros(self.n_out), trainable=True))

    # Define the neuron operations
    # input layer
    layer = 0
    self.x = tf.placeholder(shape=(None, self.n_in),dtype=tf.float64)
    self.h.append(tf.nn.relu(tf.matmul(self.x, self.W[layer]) + self.b[layer]))
    layer += 1

    # create resnet blocks
    if self.n_resnet_blocks != 0:
      for block in range(self.n_resnet_blocks):
        print('Activations of block: ', block)
        print('--activation at layer: ', layer)
        self.h.append(tf.nn.relu(tf.matmul(self.h[layer-1], self.W[layer]) + self.b[layer]))
        layer += 1
        print('--activation at layer: ', layer)
        self.h.append(tf.nn.relu(tf.matmul(self.h[layer-1], self.W[layer]) + self.b[layer] + self.h[layer-2]))
        layer += 1

    # output activation
    #if n_layer == 0:
    #  self.z = tf.nn.softmax(tf.matmul(self.x, self.W[n_layer]) + self.b[n_layer])
    #else:

    self.z = tf.nn.softmax(tf.matmul(self.h[layer-1], self.W[layer]) + self.b[layer])

    # labels
    self.z_ = tf.placeholder(shape=(None, self.n_out),dtype=tf.float64)

    #loss
    self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.z_ * tf.log(self.z), reduction_indices=[1]))
    #self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.z, labels = self.z_)) 

import tensorflow as tf
import numpy as np
import numpy.random as rd

# ResNetModel
class RnnModel():
  def __init__(self, n_symbols, n_hidden, n_out, rnn_unit, max_sequence_length,  n_layer):
    self.name = 'RNN_Model_' + rnn_unit 
    self.n_symbols = n_symbols
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.n_layer = n_layer 
    self.cell_type = rnn_unit
    self.max_sequence_length = max_sequence_length

    self.w = []
    self.b = []


    print('---Create RNN Model--- ')
    seq_length = tf.placeholder(tf.int32, [None])
    X = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.n_symbols])
    #labels
    self.z_ = tf.placeholder(tf.float32, [None, self.n_symbols])

    # define recurrent layer
    if self.cell_type == 'simple':
      cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
      # cell = tf.keras.layers.SimpleRNNCell(num_hidden) #alternative
    elif self.cell_type == 'lstm':
      cell = tf.nn.rnn_cell.LSTMCell(n_hidden)
    elif self.cell_type == 'gru':
      cell = tf.nn.rnn_cell.GRUCell(n_hidden)
    else:
      raise ValueError('bad cell type.')

    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, 7)

    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_length) # NEW
    last_outputs = outputs[:,-1,:]

    # add output neuron
    z_dim = int(self.z_.shape[1])
    self.w = tf.Variable(tf.truncated_normal([n_hidden, z_dim]))
    self.b = tf.Variable(tf.constant(.1, shape=[z_dim]))

    self.z = tf.nn.xw_plus_b(last_outputs, self.w, self.b)

    # define loss, minimizer and error
    self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.z, labels=self.z_)


    # This is incorporated in the Trainer
    #correct_prediction = tf.equal(y, tf.maximum(tf.sign(y_pred), 0))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
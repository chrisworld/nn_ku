import tensorflow as tf
import numpy as np
import numpy.random as rd

# ResNetModel
class RnnModel():
  def __init__(self, n_symbols, n_hidden, n_out, n_layer=1, cell_type='simple'):
    self.name = 'RNN_Model_' + rnn_unit 
    self.n_symbols = n_symbols
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.n_layer = n_layer 
    self.cell_type = rnn_unit

    print('---Create RNN Model--- ')
    seq_length = tf.placeholder(tf.int32, [None])
    X = tf.placeholder(tf.float32, [None, sequence_length, self.n_symbols])
    y = tf.placeholder(tf.float32, [None, self.n_symbols])

    # define recurrent layer
    if self.cell_type == 'simple':
      cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)
      # cell = tf.keras.layers.SimpleRNNCell(num_hidden) #alternative
    elif self.cell_type == 'lstm':
      cell = tf.nn.rnn_cell.LSTMCell(num_hidden)
    elif self.cell_type == 'gru':
      cell = tf.nn.rnn_cell.GRUCell(num_hidden)
    else:
      raise ValueError('bad cell type.')

    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, sequence_length=seq_length) # NEW
    last_outputs = outputs[:,-1,:]

    # add output neuron
    y_dim = int(y.shape[1])
    w = tf.Variable(tf.truncated_normal([num_hidden, y_dim]))
    b = tf.Variable(tf.constant(.1, shape=[y_dim]))

    y_pred = tf.nn.xw_plus_b(last_outputs, w, b)

    # define loss, minimizer and error
    self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y)
    self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    mistakes = tf.not_equal(y, tf.maximum(tf.sign(y_pred), 0))
    self.error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

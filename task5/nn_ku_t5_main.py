import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import numpy.random as rd
import logging
import random
import tensorflow as tf


# State transition table
TRANSITIONS = [
    [('T', 1), ('P', 2)],  # 0=B
    [('X', 3), ('S', 1)],  # 1=BT
    [('V', 4), ('T', 2)],  # 2=BP
    [('X', 2), ('S', 5)],  # 3=BTX
    [('P', 3), ('V', 5)],  # 4=BPV
    [('E', -1)],  # 5=BTXS
]

TRANSITIONS_EMB = [
    [('T', 1), ('P', 11)],  
    [('B', 1), ('P', 11)],  
    [('X', 3), ('S', 1)],  
    [('V', 4), ('T', 2)],  
    [('X', 2), ('S', 5)],  
    [('P', 3), ('V', 5)],  
    [('E', -1)],  
]

# Symbol encoding
SYMS = {'T': 0, 'P': 1, 'X': 2, 'S': 3, 'V': 4, 'B': 5, 'E': 6}

def make_reber():
    """Generate one string from the Reber grammar."""
    idx = 0
    out = 'B'
    while idx != -1:
        ts = TRANSITIONS[idx]
        symbol, idx = random.choice(ts)
        out += symbol
    return out


def make_embedded_reber():
    """Generate one string from the embedded Reber grammar."""
    c = random.choice(['T', 'P'])
    return 'B%s%s%sE' % (c, make_reber(), c)


def str_to_vec(s):
    """Convert a (embedded) Reber string to a sequence of unit vectors."""
    a = np.zeros((len(s), len(SYMS)))
    for i, c in enumerate(s):
        a[i][SYMS[c]] = 1
    return a


def str_to_next(s):
    """Given a Reber string, return a vectorized sequence of next chars.
    This is the target output of the Neural Net."""
    out = np.zeros((len(s), len(SYMS)))
    idx = 0
    for i, c in enumerate(s[1:]):
        ts = TRANSITIONS[idx]
        for next_c, _ in ts:
            out[i, SYMS[next_c]] = 1

        next_idx = [j for next_c, j in ts if next_c == c]
        assert len(next_idx) == 1
        idx = next_idx[0]

    return out

def str_to_next_embed(s):
    """Given an embedded Reber string, return a vectorized sequence of next chars.
       This is the target output of the Neural Net for embedded Reber."""
   
    out_reb = str_to_next(s[2:-2])
    ns = out_reb.shape[1]
    out = np.zeros((out_reb.shape[0]+4,ns))
    out[0,SYMS['T']]=1
    out[0,SYMS['P']]=1
    out[1,SYMS['E']]=1
    for i, st in enumerate(out_reb):
        out[i+2,:] = st
    idx = i+2
    out[idx,SYMS[s[1]]]=1
    out[idx+1,SYMS['E']]=1
    return out
    
    
def vec_to_str(xs):
    """Given a matrix, return a Reber string (with choices)."""
    idx_to_sym = dict((v,k) for k,v in SYMS.iteritems())
    out = ''
    for i in range(0, xs.shape[0]):
        vs = np.nonzero(xs[i,:])[0]
        chars = [idx_to_sym[v] for v in vs]
        if len(chars) == 1:
            out += chars[0]
        else:
            out += '{%s}' % ','.join(chars)
    return out



# Reber Batches class
class ReberBatches():
  def __init__(self, n_train_samples, n_val_samples, n_test_samples, batch_size, sym_size=7):
    self.n_train_samples = n_train_samples
    self.n_val_samples = n_val_samples
    self.n_test_samples = n_test_samples
    self.batch_size = batch_size
    self.sym_size = sym_size
    self.max_seq_len = 0
    
    # always has validation set
    self.is_validation = True

    # lists
    examples_train_list = []
    seq_len_train_list = []
    targets_train_list = []
    examples_val_list = []
    targets_val_list = []
    examples_test_list = []
    targets_test_list = []

    # max length of sequence
    self.train_max_seq_len = 0
    self.val_max_seq_len = 0
    self.test_max_seq_len = 0

    # create list of training examples
    for n in range(n_train_samples):
        example = make_embedded_reber()
        label = str_to_vec(example)
        target = str_to_next_embed(example)
        example_len = len(example)
        seq_len_train_list.append(example_len)
        if self.train_max_seq_len < example_len:
            self.train_max_seq_len = example_len
        examples_train_list.append(label)
        targets_train_list.append(target)

    # create list of validation examples
    for n in range(n_val_samples):
        example = make_embedded_reber()
        label = str_to_vec(example)
        target = str_to_next_embed(example)
        example_len = len(example)
        if self.val_max_seq_len < example_len:
            self.val_max_seq_len = example_len
        examples_val_list.append(label)
        targets_val_list.append(target)

        # create list of test examples
    for n in range(n_test_samples):
        example = make_embedded_reber()
        label = str_to_vec(example)
        target = str_to_next_embed(example)
        example_len = len(example)
        if self.test_max_seq_len < example_len:
            self.test_max_seq_len = example_len
        examples_test_list.append(label)
        targets_test_list.append(target)

    # max seq length of all reber strings
    self.max_seq_len = max(self.train_max_seq_len, self.val_max_seq_len, self.test_max_seq_len)

    # create 0-padded numpy matrix for train
    self.examples_train = np.zeros((n_train_samples, self.max_seq_len, sym_size))
    self.seq_len_train = np.zeros(n_train_samples)
    self.targets_train = np.zeros((n_train_samples, self.max_seq_len, sym_size))
    for sample_idx in range(n_train_samples):
        self.seq_len_train[sample_idx] = self.max_seq_len
        for str_idx in range(examples_train_list[sample_idx].shape[0]):
            self.examples_train[sample_idx][str_idx] = examples_train_list[sample_idx][str_idx]
            self.targets_train[sample_idx][str_idx] = targets_train_list[sample_idx][str_idx]

    # create 0-padded numpy matrix for validation
    self.examples_val = np.zeros((n_val_samples, self.max_seq_len, sym_size))
    self.seq_len_val = np.zeros(n_val_samples)
    self.targets_val = np.zeros((n_val_samples, self.max_seq_len, sym_size))
    for sample_idx in range(n_val_samples):
        self.seq_len_val[sample_idx] = self.max_seq_len
        for str_idx in range(examples_val_list[sample_idx].shape[0]):
            self.examples_val[sample_idx][str_idx] = examples_val_list[sample_idx][str_idx]
            self.targets_val[sample_idx][str_idx] = targets_val_list[sample_idx][str_idx]

    # create 0-padded numpy matrix for test
    self.examples_test = np.zeros((n_test_samples, self.max_seq_len, sym_size))
    self.seq_len_test = np.zeros(n_test_samples)
    self.targets_test = np.zeros((n_test_samples, self.max_seq_len, sym_size))
    for sample_idx in range(n_test_samples):
        self.seq_len_test[sample_idx] = self.max_seq_len
        for str_idx in range(examples_test_list[sample_idx].shape[0]):
            self.examples_test[sample_idx][str_idx] = examples_test_list[sample_idx][str_idx]
            self.targets_test[sample_idx][str_idx] = targets_test_list[sample_idx][str_idx]

    # create training batches 
    # number of batches corresponding to batch_size
    self.batch_num = np.ceil(self.examples_train.shape[0] / batch_size)
    self.batch_num_val = np.ceil(self.examples_val.shape[0] / batch_size)
    self.batch_num_test = np.ceil(self.examples_test.shape[0] / batch_size)

    # split all examples and classes
    self.batch_examples_train = np.array_split(self.examples_train, self.batch_num)
    self.batch_target_train = np.array_split(self.targets_train, self.batch_num)

    self.batch_seq_len_train = np.array_split(self.seq_len_train, self.batch_num)
    self.batch_seq_len_val = np.array_split(self.seq_len_val, self.batch_num_val)
    self.batch_seq_len_test = np.array_split(self.seq_len_test, self.batch_num_test)



# Error Collector Class
class ErrorCollector():
  def __init__(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []
    self.convergence_time_list = []
  
  def resetErrors(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []

  def addConvergenceTime(self, epoch):
    #test_error_array = np.array(self.test_error_list)
    #epoch_of_convergence = np.nonzero(test_error_array<1e-7)[0][0] + 1
    #self.convergence_time_list.append(epoch_of_convergence)
    self.convergence_time_list.append(epoch)

  def convergenceTimeMeanAndStd(self):
    convergence_time_mean = np.mean(self.convergence_time_list)
    convergence_time_std = np.std(self.convergence_time_list)
    print("-----ModelTester Convergence Time of Models-----")
    print("convergence time in epochs\nmean:                 [%.2f]\n standard deviation:  [%.2f]" %(convergence_time_mean,convergence_time_std))
    logging.info("convergence time in epochs\nmean:                 [%.2f]\n standard deviation:  [%.2f]" %(convergence_time_mean,convergence_time_std))

    return convergence_time_mean, convergence_time_std



  def addTrainError(self, train_error):
    self.train_error_list.append(train_error)

  def addTestError(self, test_error):   
    self.test_error_list.append(test_error)

  def addTrainAcc(self, train_acc):
    self.train_acc_list.append(1-train_acc) # actually it's misclassification

  def addTestAcc(self, test_acc):
    self.test_acc_list.append(1-test_acc)

  def plotTrainTestError(self, model, batch_size, learning_rate, epochs, plot_id):
    print("Plot Errors")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_error_list, color='blue', label='training', lw=2)
    ax.plot(self.test_error_list, color='red', label='test/validation', lw=1.7, linestyle= '--',alpha=0.9)
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy loss')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'plots' + os.sep
    save_name = 'Loss_' + model.name + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_opt-'+ str(model.optimizer_name) + '_lr-' + str(learning_rate) + '_id-' + str(plot_id) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()

  def plotTrainTestAcc(self, model, batch_size, learning_rate, epochs):
    print("Plot Misclassification")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_acc_list, color='blue', label='training', lw=2)
    ax.plot(self.test_acc_list, color='green', label='test', lw=2)

    ax.set_autoscaley_on(False)
    ax.set_ylim([0, 1])
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Misclassification')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'plots' + os.sep
    save_name = 'Misclass_'+ model.name + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()



# ResNetModel
class RnnModel():
  def __init__(self, n_symbols, n_hidden, n_out, rnn_unit, max_sequence_length,  n_layer, adam_optimizer):
    self.name = 'RNN_Model_' + rnn_unit 
    self.n_symbols = n_symbols
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.n_layer = n_layer 
    self.cell_type = rnn_unit
    self.max_sequence_length = max_sequence_length
    self.adam_optimizer = adam_optimizer

    # Set optimizer
    if adam_optimizer == True:
      self.optimizer_name = 'Adam'
    else:
      self.optimizer_name = 'GD'

    print('---Create ' + self.name + '---')
    self.seq_length = tf.placeholder(tf.int32, [None])
    self.X = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.n_symbols])
    #labels
    self.z_ = tf.placeholder(tf.float32, [None, self.max_sequence_length, self.n_symbols])

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

    cell = tf.contrib.rnn.OutputProjectionWrapper(cell, self.n_out, reuse=tf.AUTO_REUSE)

    outputs, states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32, sequence_length=self.seq_length)
    self.last_outputs = outputs[:,-1,:]

    # define loss, minimizer and error
    self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=self.z_)

    self.mistakes = tf.not_equal(self.z_, tf.maximum(tf.sign(outputs), 0))
    self.error = tf.reduce_mean(tf.cast(self.mistakes, tf.float32))

    # This is incorporated in the Trainer
    #correct_prediction = tf.equal(y, tf.maximum(tf.sign(y_pred), 0))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# Trainer class
class Trainer():
  def __init__(self, model, batches, error_collector):
    self.model = model
    self.batches = batches
    self.error_collector = error_collector
    self.best_validation_loss = 1000
    self.best_validation_acc = 100
    self.best_epoch = 0

    self.save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'tmp' + os.sep
    self.file_name = ""

  def train(self, learning_rate, epochs, adam_optimizer=True, early_stopping=False, early_stop_lim=1000):
    # save parameter file name
    self.file_name = 'Param_' + self.model.name + '_ep-' + str(epochs) + '_hidu-' + str(self.model.n_hidden) + '_hidl-' + str(self.model.n_layer) + '_lr-' + str(learning_rate) + '.ckpt'
    
    # setup training
    if adam_optimizer == True:
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.model.cross_entropy)
      self.optimizer_name = 'Adam'
    else:
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.model.cross_entropy)
      self.optimizer_name = 'Gradient_Descent'



    # init variables
    init = tf.global_variables_initializer() 
    # save variables
    saver = tf.train.Saver()

    # logging infos
    print("-----Training-----")
    print('Model: ' + self.model.name + ', Optimizer: ' + self.optimizer_name + ', Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))
    logging.info("-----Training-----")
    logging.info('Model: ' + self.model.name + ', Optimizer: ' + self.optimizer_name + ', Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))

    early_stop_counter = 0
    with tf.Session() as sess:
      sess.run(init)

      # run epochs
      for k in range(epochs):
        # gradient over each batch
        for batch_example, batch_class, batch_seq_len in zip(self.batches.batch_examples_train, self.batches.batch_target_train, self.batches.batch_seq_len_train):
          # training step
          sess.run(train_step, feed_dict={self.model.X: batch_example, self.model.z_: batch_class, self.model.seq_length: batch_seq_len})
        
        # Compute the errors and acc over the training dataset

        train_err = sess.run(self.model.error, feed_dict={self.model.X: self.batches.examples_train, self.model.z_: self.batches.targets_train, self.model.seq_length: self.batches.seq_len_train})
        test_loss = sess.run(self.model.error, feed_dict={self.model.X: self.batches.examples_val,
                                                          self.model.z_: self.batches.targets_val,
                                                          self.model.seq_length: self.batches.seq_len_val})
        # add errors to error collector
        self.error_collector.addTrainError(train_err)
        self.error_collector.addTestError(test_loss)

        # Early stopping, save best parameters
        if self.batches.is_validation == True and early_stopping == True:
          if test_loss < self.best_validation_loss:
          #if self.best_validation_acc == 100 or test_acc > self.best_validation_acc:
            #print("---Model saved: %s" % self.save_path + self.file_name)
            saver.save(sess, self.save_path + self.file_name)
            self.best_validation_loss = test_loss
            #self.best_validation_acc = test_acc
            self.best_epoch = k
            early_stop_counter = 0
          else:
            early_stop_counter += 1

          # stop the training if no improvement
          if early_stop_counter > early_stop_lim: 
            print("---end due to early stopping limit")
            logging.info("---end due to early stopping limit")
            break

        # logging iterations
        print("Iteration: ",k, " train loss: [%.4f]" % train_err, " valid loss: [%.4f]" % test_loss)
        logging.info("Iteration: %i" % k + " train loss: [%.4f]" % train_err + " valid loss: [%.4f]" % test_loss)
      
      if self.batches.is_validation == False or early_stopping == False:
        saver.save(sess, self.save_path + self.file_name)

      #self.error_collector.addConvergenceTime()
      self.error_collector.addConvergenceTime(self.best_epoch)

    # finish log
    print("-----Training finished with best validation loss: [%.4f] at epoch:[%i]" %(self.best_validation_loss, self.best_epoch) )
    logging.info("-----Training finished with best validation loss: [%.4f] at epoch:[%i]" %(self.best_validation_loss, self.best_epoch ) )

  # returns the path of saved model
  def getSaveFilePath(self):
    return self.save_path + self.file_name

  def resetBestScore(self):
    self.best_validation_loss = 1000
    self.best_validation_acc = 0
    self.best_epoch = 0



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




# Model Tester class
class RnnModelTester():
  def __init__(self, epochs, learning_rates, n_layer, n_hidden, n_symbols=7,
               n_out=7, rnn_unit='lstm', adam_optimizer=True):
    self.epochs = epochs
    self.learning_rates = learning_rates
    self.n_hidden = n_hidden
    self.n_layer = n_layer
    self.models = []
    self.n_out = n_out
    self.best_test_loss = 0
    self.best_conv_time = 100000
    self.best_model_param = "None"
    self.n_symbols = n_symbols
    self.rnn_unit = rnn_unit
    self.adam_optimizer = adam_optimizer

  def run(self, batches):
    # training and validation error collector
    ec = ErrorCollector()

    print("-----ModelTester-----")
    logging.info("-----ModelTester-----")

    # create models
    for n_hidden in self.n_hidden:
      for n_layer in self.n_layer:
        for rnn_unit in self.rnn_unit:
            print("RNN with max seq len: ", batches.max_seq_len)
            self.models.append(RnnModel(self.n_symbols, n_hidden, self.n_out,
                                        rnn_unit, batches.max_seq_len, n_layer, self.adam_optimizer))
    
    # training and evaluation
    plot_id = 0
    for model in self.models:
      trainer = Trainer(model, batches, ec)
      for learning_rate in self.learning_rates:
        plot_id += 1
        trainer.train(learning_rate, self.epochs, adam_optimizer=self.adam_optimizer, early_stopping=True, early_stop_lim=10)
        # print error plots
        ec.plotTrainTestError(model, batches.batch_size, learning_rate, self.epochs, plot_id)
        ec.resetErrors()
        evaluator = Evaluator(model, batches, trainer.getSaveFilePath())
        test_loss = evaluator.eval()

        # get best conversion time
        if trainer.best_epoch < self.best_conv_time:
          self.best_conv_time = trainer.best_epoch
          self.best_model_param = 'Param_' + model.name + '_ep-' + str(self.epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '_id-' + str(plot_id)

        # reset scores
        trainer.resetBestScore()

    # print convergence times
    ec.convergenceTimeMeanAndStd()

    print("-----ModelTester finished, best test error: [%.6f] and conv time (epochs): [%i] with model: %s " % (self.best_test_loss, self.best_conv_time, self.best_model_param))
    logging.info("-----ModelTester finished, best test error: [%.6f] and conv time (epochs): [%i] with model: %s " % (self.best_test_loss, self.best_conv_time, self.best_model_param))



# Main function
if __name__ == '__main__':

  # setup logging 
  log_file_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'logs' + os.sep
  log_file_name = "Model.log"
  if not os.path.exists(log_file_path):
      os.makedirs(log_file_path)
  logging.basicConfig(filename=log_file_path + log_file_name, level=logging.INFO)

  # Parameters and model
  epochs = 100
  learning_rates = [1e-2]
  n_hidden = [14] # number of hidden units within layer
  n_layer = [1]   # number of hidden layers
  rnn_unit = ['lstm', 'simple']
  adam_optimizer = True

  # Get Grammar data
  batch_size = 40
  n_train_samples = 5000
  n_val_samples = 500
  n_test_samples = 500

  # Create Reber Batches
  reber_batches = ReberBatches(n_train_samples, n_val_samples, n_test_samples, batch_size)

  # Model Testing
  model_tester = RnnModelTester(epochs, learning_rates, n_layer, n_hidden, rnn_unit=rnn_unit, adam_optimizer=adam_optimizer)
  model_tester.run(reber_batches)



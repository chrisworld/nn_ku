import numpy as np
import numpy.random as rd
import tensorflow as tf

import matplotlib.pyplot as plt
import logging
import os

# load data function
def load_isolet():
  # Loads the isolet dataset
  # Returns:
  # X....feature vectors (training set), X[i,:] is the i-th example
  # C....target classes
  # X_tst...feature vectors (test set)
  # C_tst...classes (test set)
  
  import pickle as pckl  # to load dataset
  import pylab as pl     # for graphics
  #from numpy import *    

  pl.close('all')   # closes all previous figures

  # Load dataset
  file_in = open('isolet_crop_train.pkl','rb')
  isolet_data = pckl.load(file_in) # Python 3
  #isolet_data = pckl.load(file_in, encoding='bytes') # Python 3
  file_in.close()
  X = isolet_data[0]   # input vectors X[i,:] is i-th example
  C = isolet_data[1]   # classes C[i] is class of i-th example

  file_in = open('isolet_crop_test.pkl','rb')
  isolet_test = pckl.load(file_in) # Python 3
  file_in.close()

  X_tst = isolet_test[0]   # input vectors X[i,:] is i-th example
  C_tst = isolet_test[1]   # classes C[i] is class of i-th example

  return (X, C, X_tst, C_tst)



# Error Collector class
class ErrorCollector():
  def __init__(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []
  
  def resetErrors(self):
    self.train_error_list = []
    self.test_error_list = []
    self.train_acc_list = []
    self.test_acc_list = []

  def addTrainError(self, train_error):
    self.train_error_list.append(train_error)

  def addTestError(self, test_error):   
    self.test_error_list.append(test_error)

  def addTrainAcc(self, train_acc):
    self.train_acc_list.append(1-train_acc) # actually it's misclassification

  def addTestAcc(self, test_acc):
    self.test_acc_list.append(1-test_acc)

  def plotTrainTestError(self, model, batch_size, learning_rate, epochs, activation='relu'):
    print("Plot Errors")
    fig, ax = plt.subplots(1)
    ax.plot(self.train_error_list, color='blue', label='training', lw=2)
    ax.plot(self.test_error_list, color='green', label='test', lw=2)
    #ax.set_title('Bla')
    ax.set_xlabel('Training epoch')
    ax.set_ylabel('Cross-entropy loss')
    plt.rc('grid', linestyle="--")
    plt.grid()
    plt.legend()
    # save
    save_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'plots' + os.sep
    save_name = 'Loss_' + model.name + '_act-'+ activation + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()

  def plotTrainTestAcc(self, model, batch_size, learning_rate, epochs,activation='relu'):
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
    save_name = 'Misclass_'+ model.name + '_act-'+ activation + '_ep-' + str(epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate) + '.png'
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    plt.savefig(save_path + save_name, dpi=150, bbox_inches='tight')
    #plt.show()



# ClassifiedBatch class
class ClassifiedBatches():
  def __init__(self, examples, classes, batch_size, is_test_set=False, is_validation=True):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    self.is_test_set = is_test_set
    self.is_validation = is_validation

    self.examples_train = examples
    self.examples_validation = examples
    self.classes_train = classes
    self.classes_validation = classes

    if self.is_test_set == True:
        return

    # return if no validation set is needed
    if self.is_validation == False:
      train_set_percent = 1.0
    else:
      train_set_percent = 0.7 

    # init Arrays if test set
    train_num =  int(round(train_set_percent*examples.shape[0]))
    print(train_num)
    
    #split into validation and training set
    self.examples_train = self.examples[0:(train_num)]
    self.examples_validation = self.examples[(train_num):]
    self.classes_train = self.classes[0:(train_num)]
    self.classes_validation = self.classes[(train_num):]

    # Batch List
    # number of batches corresponding to batch_size
    self.batch_num = np.ceil(self.examples_train.shape[0] / batch_size)

    # split all examples and classes
    self.batch_examples_train = np.array_split(self.examples_train, self.batch_num)
    self.batch_classes_train = np.array_split(self.classes_train, self.batch_num)  




# Batch Normalizer class
class BatchNormalizer():
  def __init__(self, examples, classes=None, batch_size=40, shuffle=False, num_classes=26):
    self.examples = examples
    self.classes = classes
    self.batch_size = batch_size
    #self.batch_num = np.round(self.examples.shape[0]/batch_size)  #number of batches corresponding to batch_size
    self.shuffle = shuffle
    self.num_classes = num_classes
    self.mean_features = np.mean(self.examples, axis=0)
    self.std_features = np.std(self.examples, axis=0)

  def shuffle_in_unison(self,examples,classes):
    assert len(examples) == len(classes)
    shuffled_a = np.empty(examples.shape, dtype=examples.dtype)
    shuffled_b = np.empty(classes.shape, dtype=classes.dtype)
    permutation = np.random.permutation(len(examples))
    for old_index, new_index in enumerate(permutation):
      shuffled_a[new_index] = examples[old_index]
      shuffled_b[new_index] = classes[old_index]
    return shuffled_a, shuffled_b

  def getMean(self):
    return self.mean_features

  def getStd(self):
    return self.std_features

  def getNormalized(self,examples, classes):
    if (self.shuffle==True):
      examples, classes = self.shuffle_in_unison(examples,classes)
    norm=(examples-(self.mean_features))/(self.std_features)
    return norm,classes

  def getBatches(self,examples,classes, test=False, is_validation=True):
    norm,norm_classes = self.getNormalized(examples,classes)
    c_oh = tf.one_hot(norm_classes-1, self.num_classes)
    with tf.Session() as sess:
      self.classes_one_hot = sess.run(c_oh)
    cbatches = ClassifiedBatches(norm, self.classes_one_hot, self.batch_size, test, is_validation)
    return cbatches



# Feedforward Model
class Model():
  def __init__(self, n_in, n_hidden, n_out, n_layer=1, activation='relu'):
    self.name = 'FeedForward_Model'
    self.n_in = n_in
    self.n_hidden = n_hidden
    self.n_out = n_out
    self.n_layer = n_layer
    self.activation = activation

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
      if self.activation == 'tanh':
        if layer == 0:
          print('Activation function of layer: ', layer, 'is', self.activation)
          self.h.append(tf.nn.tanh(tf.matmul(self.x, self.W[0]) + self.b[0]))
        else:
          print('Activation function of layer: ', layer, 'is', self.activation)
          self.h.append(tf.nn.tanh(tf.matmul(self.h[layer-1], self.W[layer]) + self.b[layer]))
      elif self.activation == 'relu':
        if layer == 0:
          print('Activation function of layer: ', layer, 'is', self.activation)
          self.h.append(tf.nn.relu(tf.matmul(self.x, self.W[0]) + self.b[0]))
        else:
          print('Activation function of layer: ', layer, 'is', self.activation)
          self.h.append(tf.nn.relu(tf.matmul(self.h[layer - 1], self.W[layer]) + self.b[layer]))
    # output activation
    if n_layer == 0:
      self.z = tf.nn.softmax(tf.matmul(self.x, self.W[n_layer]) + self.b[n_layer])
    else:
      self.z = tf.nn.softmax(tf.matmul(self.h[n_layer-1], self.W[n_layer]) + self.b[n_layer])

    # labels
    self.z_ = tf.placeholder(shape=(None, self.n_out),dtype=tf.float64)

    #loss
    self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.z_ * tf.log(self.z), reduction_indices=[1]))
    #self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.z, labels = self.z_)) 




# ResNetModel
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




# Evaluator class
class Evaluator():
  def __init__(self, model, batches, save_path):
    self.model = model
    self.batches = batches
    self.save_path = save_path

  def eval(self):

    # evaluation
    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

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



# Trainer class
class Trainer():
  def __init__(self, model, batches, error_collector):
    self.model = model
    self.batches = batches
    self.error_collector = error_collector
    self.best_validation_loss = 0
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
      optimizer_name = 'Adam'
    else:
      train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.model.cross_entropy)
      optimizer_name = 'Gradient Descent'

    correct_prediction = tf.equal(tf.argmax(self.model.z,1), tf.argmax(self.model.z_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

    # init variables
    init = tf.global_variables_initializer() 
    # save variables
    saver = tf.train.Saver()

    # logging infos
    print("-----Training-----")
    print('Model: ' + self.model.name + ', Optimizer: ' + optimizer_name + ', Activation: ' + self.model.activation +  ', Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))
    logging.info("-----Training-----")
    logging.info('Model: ' + self.model.name + ', Optimizer: ' + optimizer_name + ', Activation: ' + self.model.activation + ', Epochs: ' + str(epochs) + ', Hidden Units: ' + str(self.model.n_hidden) + ', HiddenLayer: ' + str(self.model.n_layer) + ', LearningRate: ' + str(learning_rate))

    early_stop_counter = 0
    with tf.Session() as sess:
      sess.run(init)

      # run epochs
      for k in range(epochs):
        # gradient over each batch
        for batch_example, batch_class in zip(self.batches.batch_examples_train, self.batches.batch_classes_train):
          # training step
          sess.run(train_step, feed_dict={self.model.x: batch_example, self.model.z_: batch_class})
        
        # Compute the errors and acc over the training dataset
        train_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples_train, self.model.z_: self.batches.classes_train})
        train_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples_train, self.model.z_: self.batches.classes_train})
        self.error_collector.addTrainError(train_loss)
        self.error_collector.addTrainAcc(train_acc)
        # Compute the errors and acc of the validation set
        test_loss = sess.run(self.model.cross_entropy, feed_dict={self.model.x: self.batches.examples_validation, self.model.z_: self.batches.classes_validation})
        test_acc = sess.run(accuracy, feed_dict={self.model.x: self.batches.examples_validation, self.model.z_: self.batches.classes_validation})
        self.error_collector.addTestError(test_loss)
        self.error_collector.addTestAcc(test_acc)

        # Early stopping, save best parameters
        if self.batches.is_validation == True and early_stopping == True:
          #if self.best_validation_loss == 0 or test_loss < self.best_validation_loss:
          if self.best_validation_acc == 100 or test_acc > self.best_validation_acc:
            #print("---Model saved: %s" % self.save_path + self.file_name)
            saver.save(sess, self.save_path + self.file_name)
            self.best_validation_loss = test_loss
            self.best_validation_acc = test_acc
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
        print("Iteration: ",k, " train loss: [%.4f]" % train_loss, " train acc: [%.4f]" % train_acc, " valid loss: [%.4f]" % test_loss, " valid acc: [%.4f]" % test_acc)
        logging.info("Iteration: %i" % k + " train loss: [%.4f]" % train_loss + " train acc: [%.4f]" % train_acc + " valid loss: [%.4f]" % test_loss + " valid acc: [%.4f]" % test_acc)
      
      if self.batches.is_validation == False or early_stopping == False:
        saver.save(sess, self.save_path + self.file_name)


    # finish log
    print("-----Training finished with best validation loss: [%.4f] validation acc: [%.4f] at epoch:[%i]" %(self.best_validation_loss, self.best_validation_acc, self.best_epoch) )
    logging.info("-----Training finished with best validation loss: [%.4f] validation acc: [%.4f] at epoch:[%i]" %(self.best_validation_loss, self.best_validation_acc, self.best_epoch ) )

  # returns the path of saved model
  def getSaveFilePath(self):
    return self.save_path + self.file_name

  def resetBestScore(self):
    self.best_validation_loss = 0
    self.best_validation_acc = 0
    self.best_epoch = 0




# Model Tester class
class ModelTester():
  def __init__(self, epochs, learning_rates, n_hidden, n_layer, n_in=300, n_out=26, activation='relu', is_res_net = False):
    self.epochs = epochs
    self.learning_rates = learning_rates
    self.n_hidden = n_hidden
    self.n_layer = n_layer
    self.models = []
    self.n_in = n_in
    self.n_out = n_out
    self.best_test_acc = 0
    self.best_model_param = "None"
    self.activation = activation
    self.is_res_net = is_res_net

  def run(self, train_batches, test_batches):
    # training and validation error collector
    ec = ErrorCollector()

    print("-----ModelTester-----")
    logging.info("-----ModelTester-----")

    for n_hidden in self.n_hidden:
      for n_layer in self.n_layer:
        if self.is_res_net:
          self.models.append(ResNetModel(n_in=self.n_in, n_hidden=n_hidden, n_out=self.n_out, n_layer=n_layer, activation='relu'))
        else:
          for activation in self.activation:
            # create models
            self.models.append(Model(n_in=self.n_in, n_hidden=n_hidden, n_out=self.n_out, n_layer=n_layer, activation=activation))
          
    
    for model in self.models:
      trainer = Trainer(model, train_batches, ec)
      for learning_rate in self.learning_rates:
        trainer.train(learning_rate, self.epochs, early_stop_lim=25)
        # print error plots
        ec.plotTrainTestError(model, train_batches.batch_size, learning_rate, self.epochs, model.activation)
        ec.plotTrainTestAcc(model, train_batches.batch_size, learning_rate, self.epochs, model.activation)
        ec.resetErrors()
        evaluator = Evaluator(model, test_batches, trainer.getSaveFilePath())
        test_loss, test_acc  = evaluator.eval()
        trainer.resetBestScore()

        if self.best_test_acc == 0 or test_acc > self.best_test_acc:
          self.best_test_acc = test_acc
          self.best_model_param = 'Param_' + model.name + '_ep-' + str(self.epochs) + '_hidu-' + str(model.n_hidden) + '_hidl-' + str(model.n_layer) + '_lr-' + str(learning_rate)

    print("-----ModelTester finished, best test acc: [%.6f] with model: %s " % (self.best_test_acc, self.best_model_param))
    logging.info("-----ModelTester finished, best test acc: [%.6f] with model: %s " % (self.best_test_acc, self.best_model_param))




# Main function
if __name__ == '__main__':

  X, C, X_tst, C_tst = load_isolet()

  # setup logging 
  log_file_path = os.path.dirname(os.path.abspath( __file__ )) +  os.sep + 'logs' + os.sep
  log_file_name = "Model.log"
  if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)
  logging.basicConfig(filename=log_file_path + log_file_name, level=logging.INFO)

  # Parameters and model
  epochs = 100
  learning_rate = [3e-5]
  n_hidden = [40] # number of hidden units within layer
  n_layer = [9]   # number of hidden layers
  activation = ['relu', 'tanh']
  batch_size = 20

  # Batch Normalizer
  bn = BatchNormalizer(X, C, batch_size=batch_size, shuffle=True)
  train_batches = bn.getBatches(X, C)
  test_batches = bn.getBatches(X_tst, C_tst, test=True)
  # use Test set as Validation
  train_batches.examples_validation = test_batches.examples_validation
  train_batches.classes_validation = test_batches.classes_validation

  model_tester = ModelTester(epochs, learning_rate, n_hidden, n_layer, activation=activation, is_res_net = False)
  model_tester.run(train_batches, test_batches)

  resnet_tester = ModelTester(epochs, learning_rate, n_hidden, n_layer, activation=activation, is_res_net = True)
  # get the best from the other model tester
  resnet_tester.best_test_acc = model_tester.best_test_acc
  resnet_tester.best_model_param = model_tester.best_model_param
  resnet_tester.run(train_batches, test_batches)


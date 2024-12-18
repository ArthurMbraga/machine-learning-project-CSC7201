import numpy as np
import random

from sklearn.utils import shuffle


import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard


import os
from os.path import isfile
from functools import partial


# The following library is to plot the loss during training
# https://github.com/stared/livelossplot
#! pip install livelossplot
from livelossplot import PlotLossesKerasTF


# To show time at each cell
# Credits to https://medium.com/@arptoth/how-to-measure-execution-time-in-google-colab-707cc9aad1c8
#!pip install ipython-autotime


def make_sequential_model(sample_size, layer_sizes,
                          hidden_activation_function="relu",
                          out_activation_function="softmax",
                          loss_fun="categorical_crossentropy",
                          learning_rate=0.01,
                          regularization_coeff=0,
                          metrics=['accuracy']):
    """
    Makes a sequential model.
    Parameters
    -------------------------
    sample_size: integer
      The number of features of the samples

    layer_sizes: list
      List of the size of the neural network layers. For instance, if
      layer_sizes = [8, 6, 4], the 1st layer will have 5 neurons, the 2nd 6 etc.
      Attention: the size of the last layer (the output layer) is not arbitrary.
      In case of monodimensional regression, it must be 1.
      When using categorical_crossentropy, it must be the same as the number of
      categories.
      When using binary_crossentropy, it must be 1.

    inner_activation_function: string
      Activation function used in all layers, except the last one.
      Ex: "relu"

    out_activation_function: string
      Activation function of the last layer.
      Ex. "softmax"

    loss_fun: string
      The loss function we want to minimize. Ex. categorical_crossentropy

    learning_rate: float
      Ex. 0.01

    regularization_coeff: float
      Coefficient of ridge regression
      Ex. 0.01

    metrics: list of strings
      The metrics we want to show during training. Ex. ['accuracy']
    """

    model = Sequential()

    # In the next code we will use `partial`, which is a function of the ptyhon
    # library functools, which allows to define a class, identical to another
    # class but with some different default values.
    # In our case we define MyDenseLayer equal to the standard keras class
    # `Dense`, which implements a simple neural network layer, specifying
    # two default values: one for the activation function, and another for the
    # regularization

    if (regularization_coeff == 0):
        # No regularization
        MyDenseLayer = partial(Dense, activation=hidden_activation_function)
    else:
        MyDenseLayer = partial(Dense, activation=hidden_activation_function,
                               kernel_regularizer=keras.regularizers.l2(regularization_coeff))

    # Add the input layer
    model.add(MyDenseLayer(layer_sizes[0],
                           input_dim=sample_size))

    # Add hidden layers
    # We iterate from the 2nd element to the penultimate
    for i in range(1, len(layer_sizes)-1):
        model.add(MyDenseLayer(layer_sizes[i]))

    # Add output layer
    model.add(Dense(layer_sizes[-1],
                    activation=out_activation_function))

    model.compile(loss=loss_fun,
                  optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=metrics)

    return model


def enforce_reproducibility(seed):
    tf.keras.backend.clear_session()

    # To know more:
    #       https://machinelearningmastery.com/reproducible-results-neural-networks-keras/
    random.seed(seed)
    np.random.seed(random.randint(0, 300000))
    tf.random.set_seed(random.randint(0, 300000))


def train_model(model, nn_file, X_tr, y_tr, seed, max_epochs=1000,
                overwrite=True, validation_split=0.2, patience=10):
    """
    model: neural network model
              It must be a compiled neural network, e.g., a model issued by the
              function make_sequential_model(..) defined before

    nn_file:  string (name of a file)
              This file will be used to store the weights of the trained neural
              network. Such weights are automatically stored during training 
              (thanks to the ModelCheckpoint callback (see the implementation 
              code)), so that even if the code fails in the middle of training,
              you can resume training without starting from scratch.
              If the file already exists, before starting training, the weights
              in such a file will be loaded, so that we do not start training from
              scratch, but we start already from (hopefully) good weigths.

    overwrite: boolean
              If true, the model will be built and trained from scratch, 
              indipendent of whether nn_file exists or not.

    seed: integer

    X_tr: matrix
              Feature matrix of the training set

    y_tr: matrix
              True labels of the training set

    max_epochs: integer
              Training will stop after such number of epochs

    validation_split: float (between 0 and 1)
              Fraction of training dataset that will be used as validation

    patience: integer
              Training will stop if the validation loss does not improve after the 
              specified number of epochs
    """

    enforce_reproducibility(seed)

    # Before starting training, Keras divides (X_tr, y_tr) into a training subset
    # and a validation subset. During iterations, Keras will do backpropagation
    # in order to minimize the loss on the trainins subset, but it will monitor
    # and also plot the loss on the validation subset.
    # However, Keras always takes the first part of (X_tr, y_tr) as training
    # subset and the second part as validation subset. This can be bad, in case
    # the dataset has been created with a certain order (for instance all the
    # samples with a certain characteristic first, and then all the others), as
    # we instead need to train the neural network on a representative subset of
    # samples. For this reason, we first shuffle the dataset
    X_train, y_train = shuffle(X_tr, y_tr, random_state=seed)

    ##################
    #### CALLBACKS ###
    ##################
    # These functions are called at every epoch
    plot_cb = PlotLossesKerasTF()  # Plots the loss
    checkpoint_cb = ModelCheckpoint(nn_file)  # Stores weights
    logger_cb = CSVLogger(nn_file+'.csv', append=True)  # Stores history
    # see https://theailearner.com/2019/07/23/keras-callbacks-csvlogger/

    # To stop early if we already converged
    # See pagg 315-16 of [Ge19]
    early_stop_cb = tf.keras.callbacks.EarlyStopping(verbose=1,
                                                     monitor='val_loss',
                                                     patience=patience, restore_best_weights=True)

    if overwrite == True:
        try:
            os.remove(nn_file)
        except OSError:
            pass

        try:
            os.remove(nn_file+'.csv')
        except OSError:
            pass

    if isfile(nn_file):
        print("Loading pre-existing model")
        model = load_model(nn_file)

    history = model.fit(X_train, y_train, epochs=max_epochs,
                        callbacks=[plot_cb, checkpoint_cb,
                                   logger_cb, early_stop_cb],
                        validation_split=validation_split)

    return history

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:34:56 2025

@author: HP
"""

"""============================================================================
                Visualize results of numerical experimenst
============================================================================"""    

from neural_network import *
import autograd.numpy as np

# for plotting results
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.mplot3d import Axes3D                 # for 3d plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd


""" compute test accuracy before and after training """


def init_train_network(data, activation_funcs, layer_output_sizes, 
                  input_size, output_size, cost_fnc, optimizer_,
                  lambda_ = 0.0, reg = None,
                  epochs_=500):
    """ initialize and train a neural network """
    # load data
    x_train, x_test, y_train, y_test = data
    activation_derivatives = get_activation_ders(activation_funcs)
    cost_der = globals()[cost_fnc.__name__ + '_der']
    nn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der, lmd=lambda_,
        regularization=reg
        )
    """ make first predictions on the train data and compute the 
    cost function """
    first_predicts = nn.predict(x_test)
    cost_first_guess = cost_fnc(first_predicts, y_test)
    # train the network
    nn.train_network(x_train, y_train, batches=10, optimizer=optimizer_, 
                     epochs=epochs_)
    return nn, cost_first_guess

def compute_test_accuracy(network, data, return_predicts=False):
    """ sometimes we also want the predictions explicitly, then we also return
    them. In general, we only need the accuracy. Hence return_predict is set
    to False in general """
    x_train, x_test, y_train, y_test = data
    predicts = network.predict(x_test)
    cost_after_training = network.cost_fnc(predicts, y_test)
    if return_predicts:
        return cost_after_training, predicts
    else:
        return cost_after_training
    
def test_accuracy(data, activation_funcs, layer_output_sizes, 
                  input_size, output_size, cost_fnc, optimizer_,
                  lambda_ = 0.0, reg = None,
                  epochs_=500, return_predicts=False):
    """ combines the two functions above - this facilitates later use, e.g.,
    when we are only interested in the test accuracy / error, we only have
    to call one function. """
    network, _ = init_train_network(data, activation_funcs, layer_output_sizes, 
                                 input_size, output_size, cost_fnc, optimizer_,
                                 lambda_, reg, epochs_)
    return compute_test_accuracy(network, data, return_predicts)

def compute_train_accuracy(network, data):
    x_train, x_test, y_train, y_test = data
    predicts = network.predict(x_train)
    cost_train = network.cost_fnc(predicts, y_train)
    return cost_train

def train_accuracy(data, activation_funcs, layer_output_sizes, 
                  input_size, output_size, cost_fnc, optimizer_,
                  lambda_ = 0.0, reg = None,
                  epochs_=500, return_predicts=False):
    network, _ = init_train_network(data, activation_funcs, layer_output_sizes, 
                                 input_size, output_size, cost_fnc, optimizer_,
                                 lambda_, reg, epochs_)
    return compute_train_accuracy(network, data)
    
def compare_train_test(data,
                  input_size, output_size, cost_fnc, optimizer_,
                  lambda_ = 0.0, reg = None):
    """
    function that computes train and test error for a neural network depending
    on the number of hidden layers.
    For simplicity, we use 50 nodes in every layer.
    """
    train_errs = []
    test_errs = []
    numbers_hidden_layers = [1, 2, 3, 4, 5, 6]
    epochs = [500, 500, 500, 600, 800, 1000]
    for i in numbers_hidden_layers:
        layer_output_sizes = [40 for _ in range(i)]
        layer_output_sizes.append(output_size)
        if i == 1:
            activation_funcs = [sigmoid]
        elif i == 2:
            activation_funcs = [ReLU, sigmoid]
        elif i == 3:
            activation_funcs = [ReLU, LeakyReLU, sigmoid]
        elif i == 4:
            activation_funcs = [ReLU, sigmoid, LeakyReLU, sigmoid]
        elif i == 5:
            activation_funcs = [ReLU, ReLU, sigmoid, LeakyReLU, sigmoid]
        elif i == 6:
            activation_funcs = [ReLU, ReLU, ReLU, LeakyReLU, LeakyReLU, sigmoid]
        elif i == 7:
            activation_funcs = [ReLU, ReLU, LeakyReLU, ReLU, LeakyReLU, 
                                LeakyReLU, sigmoid]
        activation_funcs.append(identity)
        train_err = 0
        test_err = 0
        if reg is not None:
            lmb = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                              input_size, output_size, cost_fnc, optimizer_,
                              reg, epochs[i-1])
        for _ in range(50):
            # compute train and test errors multiple times and take the mean
            # at the end
            network, _ = init_train_network(data, activation_funcs, layer_output_sizes, 
                                        input_size, output_size, cost_fnc, optimizer_,
                                        lmb, reg, epochs[i-1])
            train_err += compute_train_accuracy(network, data)
            test_err += compute_test_accuracy(network, data)
        train_errs.append(train_err / 50)
        test_errs.append(test_err / 50)
    return train_errs, test_errs

def plot_train_test_errs(data,
                  input_size, output_size, cost_fnc, optimizer_,
                  lambda_ = 0.0, reg = None):
    """ plot results from previous function """
    x, y = compare_train_test(data, input_size, output_size, cost_fnc, optimizer_,
                              lambda_, reg)
    plt.figure(dpi=150)
    plt.plot(np.arange(1, len(x) + 1), x, color='red', marker='D', label='train error')
    plt.plot(np.arange(1, len(y) + 1), y, color='blue', marker='D', label='test error')
    plt.xlabel('number hidden layers')
    plt.ylabel('error')
    plt.grid()
    plt.legend()
    plt.show()
    
    

""" computes test accuracy for networks with different numbers of hidden
layers and numbers of nodes per layer """

def test_different_layers(data, number_hidden_layers, nodes_per_layer,
                          input_size, output_size, cost_fnc, optimizer_,
                          reg = None, epochs=500):
    accuracies = []
    for i in number_hidden_layers:
        accuracy_i = []
        activation_funcs = [sigmoid for _ in range(i+1)]
        for j in nodes_per_layer:
            layer_output_sizes = [j for _ in range(i)]
            layer_output_sizes.append(output_size)
            if reg is not None:
                # tune parameter
                lmb = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                                  input_size, output_size, cost_fnc, optimizer_,
                                  reg, epochs_=epochs)
                print("lambda = ", lmb)    # only for testing
            else:
                lmb = 0.0
            np.random.seed(123)
            acc = test_accuracy(data, activation_funcs, layer_output_sizes, 
                                   input_size, output_size, cost_fnc, optimizer_,
                                   lambda_ = lmb, reg = reg,
                                   epochs_=epochs)
            accuracy_i.append(acc)
        accuracies.append(accuracy_i)
    return accuracies

""" plot a heat map which shows the test accuracy depending on the number of
hidden layers and number of nodes per layer """

def heat_map_test_accuracy(data, number_hidden_layers, nodes_per_layer,
                           input_size, output_size, cost_fnc, optimizer_,
                           reg = None, epochs=500):
    values = test_different_layers(data, number_hidden_layers, nodes_per_layer,
                               input_size, output_size, cost_fnc, optimizer_,
                               reg = reg, epochs=epochs)
    k, l = len(number_hidden_layers), len(nodes_per_layer)
    z = np.array(values).reshape((k, l))
    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(z, cmap=cm.jet)
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(range(l), labels=nodes_per_layer)
    # y_labels = np.array(['10^{}'.format(int(i)) for i in np.log10(params)])
    ax.set_yticks(range(k), labels=number_hidden_layers)
    ax.set_ylabel('number of hidden layers')
    ax.set_xlabel('nodes per layer')
    # make colorbar fit to size of the heat map
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.show()


""" plot of a simple approximation of a 1D function """

def plot_1D_approx(x_data, y_true, y_predict):
    plt.figure(dpi=150)
    plt.scatter(x_data, y_true, label='test data')
    plt.scatter(x_data, y_predict, label='predictions')
    plt.grid()
    plt.legend()
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.show()
    
    
"""============================================================================
                        tuning of parameters
============================================================================"""

### find optimal regularization parameter

def try_reg_parameters(data, activation_funcs, layer_output_sizes, 
                  input_size, output_size, cost_fnc, optimizer_,
                  regularization, lambdas, epochs=500):
    lst_results = []
    for lmb in lambdas:
        try:
            # set a seed such that results are comparable
            np.random.seed(123)
            err = test_accuracy(data, activation_funcs, layer_output_sizes, 
                                input_size, output_size, cost_fnc, optimizer_,
                                lambda_ = lmb, reg = regularization,
                                epochs_=epochs)
        except:
            err = np.inf
        lst_results.append(err) 
    return lambdas[np.argmin(np.array(lst_results))], np.min(np.array(lst_results))

def optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                  input_size, output_size, cost_fnc, optimizer_,
                  regularization, epochs_=500):
    """ first search the best parameter in 10^{-i}, i=0,...,6. Afterwards, search
    for better parameters in a neighborhood of the current optimum """
    lambdas = np.logspace(-6, 0, 7)
    lmb_, min_ = try_reg_parameters(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer_,
                      regularization, lambdas, epochs=epochs_)
    new_params1 = np.linspace(0.1 * lmb_, 0.9 * lmb_, 9)
    lmb_1, min_1 = try_reg_parameters(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer_,
                      regularization, new_params1, epochs=epochs_)
    new_params2 = np.linspace(1.5 * lmb_, 5 * lmb_, 8)
    lmb_2, min_2 = try_reg_parameters(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer_,
                      regularization, new_params2, epochs=epochs_)
    dict_ = {min_ : lmb_, min_1 : lmb_1, min_2 : lmb_2}
    # return the minimal value
    return dict_[np.min(list(dict_))]
    
def reg_parameters_network_depth(data, input_size, output_size, cost_fnc, 
                                 optimizer_, regularization, epochs=500):
    layer_sizes_lst = [[20, 20], [50, 50], [50, 100], [100, 100],
                       [50, 50, 50], [50, 50, 100], [100, 100, 100],
                       [50, 50, 50, 50], [50, 100, 100, 50]]
    optimal = []
    for layer_sizes in layer_sizes_lst:
        print(layer_sizes)
        if len(layer_sizes) == 2:
            activation_funcs = [ReLU, sigmoid]
        elif len(layer_sizes) == 3:
            activation_funcs = [ReLU, LeakyReLU, sigmoid]
        elif len(layer_sizes) == 4:
            activation_funcs = [ReLU, sigmoid, LeakyReLU, sigmoid]
        layer_sizes.append(output_size)
        activation_funcs.append(identity)
        optimal_lmbd = optimal_reg_parameter(data, activation_funcs, layer_sizes, 
                                             input_size, output_size, cost_fnc, 
                                             optimizer_, regularization)
        optimal.append(optimal_lmbd)
    df = pd.DataFrame(layer_sizes_lst, optimal)
    return df
        
        
        
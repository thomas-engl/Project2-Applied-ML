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
from data_sets import *

# for plotting results
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.mplot3d import Axes3D                 # for 3d plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

import pandas as pd
import time


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
                  input_size, output_size, cost_fnc, optimizer,
                  lambda_ = 0.0, reg = None):
    """
    function that computes train and test error for a neural network depending
    on the number of hidden layers.
    For simplicity, we use 50 nodes in every layer.
    """
    train_errs = []
    test_errs = []
    numbers_hidden_layers = np.arange(1, 9)
    epochs = [500 for _ in range(8)]
    for i in numbers_hidden_layers:
        layer_output_sizes = [50 for _ in range(i)]
        layer_output_sizes.append(output_size)
        activation_funcs = [sigmoid for _ in range(i)]
        activation_funcs.append(identity)
        train_err = 0
        test_err = 0
        eta_opt = tune_learning_rate(data, activation_funcs, layer_output_sizes, 
                                     input_size, output_size, cost_fnc, optimizer)
        if reg is not None:
            lmb = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                              input_size, output_size, cost_fnc, optimizer(
                              eta=eta_opt), reg, epochs[i-1])
        else:
            lmb = 0.0
        n_tests = 5
        for j in range(n_tests):
            # compute train and test errors multiple times and take the mean
            # at the end
            np.random.seed(j)
            network, _ = init_train_network(data, activation_funcs, layer_output_sizes, 
                                        input_size, output_size, cost_fnc, optimizer(
                                        eta=eta_opt), lmb, reg, epochs[i-1])
            train_err += compute_train_accuracy(network, data)
            test_err += compute_test_accuracy(network, data)
        print("number layers: ", i)
        print("average test MSE  = ", test_err / n_tests)
        print("average train MSE = ", train_err / n_tests)
        train_errs.append(train_err / n_tests)
        test_errs.append(test_err / n_tests)
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
    # if we use regularization we also want to know the values of the regula-
    # rization hyper parameters
    if reg is not None:
        lambdas = []
    for i in number_hidden_layers:
        accuracy_i = []
        if i == 0:
            activation_funcs = []
        elif i == 1:	
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
        activation_funcs.append(identity)
        for j in nodes_per_layer:
            layer_output_sizes = [j for _ in range(i)]
            layer_output_sizes.append(output_size)
            if reg is not None:
                # tune regularization hyper parameter lambda
                # use 30 % of the actual number of epochs for the tuning
                lmb = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                                  input_size, output_size, cost_fnc, optimizer_,
                                  reg, epochs_=int(round(0.4 * epochs)))
                print(lmb)
                lambdas.append(lmb)
            else:
                lmb = 0.0
            # use a seed such that results are comparable
            np.random.seed(123)
            acc = test_accuracy(data, activation_funcs, layer_output_sizes, 
                                   input_size, output_size, cost_fnc, optimizer_,
                                   lambda_ = lmb, reg = reg,
                                   epochs_=epochs)
            accuracy_i.append(acc)
        accuracies.append(accuracy_i)
    if reg is not None:
        return accuracies, lambdas
    else:
        return accuracies

""" plot a heat map which shows the test accuracy depending on the number of
hidden layers and number of nodes per layer """

def heat_map_test_accuracy(data, number_hidden_layers, nodes_per_layer,
                           input_size, output_size, cost_fnc, optimizer_,
                           reg = None, epochs=500):
    # create a list with results from the above function so that we can
    # iterate over it afterwards, even if it consists of only one element
    if reg is None:
        values = [test_different_layers(data, number_hidden_layers, 
                            nodes_per_layer, input_size, output_size, cost_fnc, 
                            optimizer_, reg = reg, epochs=epochs)]
    else:
        acc, lmb = test_different_layers(data, number_hidden_layers, 
                            nodes_per_layer, input_size, output_size, cost_fnc, 
                            optimizer_, reg = reg, epochs=epochs)
        values = [acc, lmb]
        
    for i in range(len(values)):
        k, l = len(number_hidden_layers), len(nodes_per_layer)
        z = np.array(values[i]).reshape((k, l))
        fig, ax = plt.subplots(dpi=200)
        if i == 0:
            im = ax.imshow(z, cmap=cm.jet)
        else:
            im = ax.imshow(z, cmap=cm.jet, norm=LogNorm(vmin=z.min(), 
                                                        vmax=z.max()))
        # Show all ticks and label them with the respective list entries
        ax.set_xticks(range(l), labels=nodes_per_layer)
        # y_labels = np.array(['10^{}'.format(int(i)) for i in np.log10(params)])
        ax.set_yticks(range(k), labels=number_hidden_layers)
        ax.set_ylabel('number of hidden layers')
        ax.set_xlabel('nodes per layer')
        # make colorbar fit to size of the heat map
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        # colorbar
        cbar = ax.figure.colorbar(im, cax=cax)
        if i == 0:
            cbar.ax.set_ylabel('test MSE', rotation=-90, va="bottom")
        else:
            cbar.ax.set_ylabel(r'regularization parameter $\lambda$', 
                               rotation=-90, va="bottom")
        # fig.tight_layout()
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
                        tuning of hyper paramaters
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
    """
    find best regularization hyper parameter via random search and one
    refinement
    """
    # sample 4 unique random integers in the range (0,...,6)
    rng = np.random.default_rng()
    samples = rng.choice(7, size=4, replace=False) 
    lambdas = np.array(10.0**(- samples), dtype=float)
    lmb_, min_ = try_reg_parameters(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer_,
                      regularization, lambdas, epochs=epochs_)
    # do one refinement around lmb_ and sample 5 random numbers between 
    # 0.5 * lmb_ to 5 lmb_ on a logarithmic scale
    params_ = np.logspace(np.log10(0.5 * lmb_), np.log10(5 * lmb_), 10)
    new_samples = rng.choice(10, size=5, replace=False) 
    new_params = params_[new_samples]
    lmb_1, min_1 = try_reg_parameters(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer_,
                      regularization, new_params, epochs=epochs_)
    dict_ = {min_ : lmb_, min_1 : lmb_1}
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

### tuning of the learning rate

def try_learning_rate(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer,
                      etas):
    """ without regularization for now """
    lst_results = []
    for eta_ in etas:
        optimize_algorithm = optimizer(eta=eta_)
        try:
            # set a seed such that results are comparable
            np.random.seed(123)
            err = test_accuracy(data, activation_funcs, layer_output_sizes, 
                                input_size, output_size, cost_fnc, 
                                optimize_algorithm, epochs_=10)
        except:
            err = np.inf
        lst_results.append(err) 
    return etas[np.argmin(np.array(lst_results))], np.min(np.array(lst_results))

def tune_learning_rate(data, activation_funcs, layer_output_sizes, 
                       input_size, output_size, cost_fnc, optimizer):
    # sample 4 unique random integers in the range (0,...,6)
    rng = np.random.default_rng()
    samples = rng.choice(7, size=4, replace=False) 
    etas = np.array(10.0**(- samples), dtype=float)
    eta_, min_ = try_learning_rate(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer, etas)
    # do one refinement around eta_ and sample 5 random numbers between 
    # 0.5 * eta_ to 5 eta_ on a logarithmic scale
    params_ = np.logspace(np.log10(0.5 * eta_), np.log10(5 * eta_), 10)
    new_samples = rng.choice(10, size=5, replace=False) 
    new_params = params_[new_samples]
    eta_1, min_1 = try_learning_rate(data, activation_funcs, layer_output_sizes, 
                      input_size, output_size, cost_fnc, optimizer,
                      new_params)
    dict_ = {min_ : eta_, min_1 : eta_1}
    # return the minimal value
    return dict_[np.min(list(dict_))]
    
    
    

"""============================================================================
                compare different norms and influence of regularization
                depending on number of nodes per layer
============================================================================"""
    
        
def compare_norms(data, input_size, output_size, cost_fnc,
                  optimizer, epochs=500):
    # check different numbers of nodes per layer
    # for a better comparison, we always work with two hidden layers and
    # ReLU and sigmoid as activation functions
    activation_funcs = [ReLU, sigmoid, identity]
    errs_no_reg = []
    errs_L1_reg = []
    errs_L2_reg = []
    for i in (20, 40, 60, 80, 100):
        layer_output_sizes = [i, i, output_size]
        errs_no_reg.append(
            test_accuracy(data, activation_funcs, layer_output_sizes, 
                                       input_size, output_size, cost_fnc, 
                                       optimizer)
            )
        # L1 regularization
        lmb1 = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                                     input_size, output_size, cost_fnc, 
                                     optimizer, 'L1', epochs_=epochs)
        errs_L1_reg.append(
            test_accuracy(
                data, activation_funcs, layer_output_sizes, input_size, 
                output_size, cost_fnc, optimizer, lambda_=lmb1, reg='L1', 
                epochs_=epochs)
            )
        # L2 regularization
        lmb2 = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                                     input_size, output_size, cost_fnc, 
                                     optimizer, 'L2', epochs_=epochs)
        errs_L2_reg.append(
            test_accuracy(
                data, activation_funcs, layer_output_sizes, input_size, 
                output_size, cost_fnc, optimizer, lambda_=lmb2, reg='L2', 
                epochs_=epochs)
            )
    return errs_no_reg, errs_L1_reg, errs_L2_reg

def plot_compare_norms(data, input_size, output_size, cost_fnc,
                       optimizer, epochs=500):
    plt.figure(dpi=150)
    x = np.array([20, 40, 60, 80, 100])
    y1, y2, y3 = compare_norms(data, input_size, output_size, cost_fnc,
                      optimizer, epochs)
    plt.semilogy(x, y1, color='red', marker='D', label='no regularization')
    plt.semilogy(x, y2, color='blue', marker='D', label=r'$L^1$ regularization')
    plt.semilogy(x, y3, color='green', marker='D', label=r'$L^2$ regularization')
    plt.xlabel('nodes per layer')
    plt.ylabel('test MSE')
    plt.grid()
    plt.legend()
    plt.show()
    
    
"""============================================================================
                        curse of dimensionality?
 test how well our neural network can approximate higher-dimensional functions
============================================================================"""

def test_dimensionality(activation_funcs, layer_output_sizes, optimizer):
    """
    tests the approximation quality of our neural network when fitting the
    d-dimensional Rastrigin function. d varies from 2 to 5. 
    In higher dimension we need more data, therefore we use 10**(2d-1) data points.
    Also, we need more epochs with increasing d. Hence, set the number of epochs
    to 400(d-1). We measure the test MSE and the run time. The network archi-
    tecture is always the same.

    Parameters
    ----------
    activation_funcs : list
        contains activation functions of the network layers
    layer_output_sizes : list
        contains output sizes of the network layers
    optimizer : class
        optimization algorithm for the network training

    Returns
    -------
    list_mses : list
        test MSEs for considered networks
    list_times : list
        runtimes to compute the test MSE

    """
    dims = np.arange(2, 6)
    list_n_points = [1_000, 10_000, 50_000, 100_000]
    list_epochs = [500, 750, 1_000, 1_500]
    list_mses = []
    list_times = []
    for d in dims:
        x_train, x_test, y_train, y_test = load_rastrigin_data(list_n_points[d-2], d)
        data = get_scaled_data(x_train, x_test, y_train, y_test)
        input_size = d
        output_size = 1
        eta_opt = tune_learning_rate(data, activation_funcs, layer_output_sizes, 
                                     input_size, output_size, mse, optimizer)
        start = time.perf_counter()
        test_mse = test_accuracy(data, activation_funcs, layer_output_sizes, 
                                 input_size, 1, mse, optimizer(eta=eta_opt), 
                                 epochs_=list_epochs[d-2])
        end = time.perf_counter()
        list_mses.append(test_mse)
        list_times.append(end - start)
        print("d        = ", d)
        print("optimal learning rate : ", eta_opt)
        print("test_mse = ", test_mse)
        print("time     = ", end - start, "\n")
    return list_mses, list_times

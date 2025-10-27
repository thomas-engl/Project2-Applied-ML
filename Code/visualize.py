# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:34:56 2025

@author: HP
"""

"""============================================================================
                Visualize results of numerical experimenst
============================================================================"""    

from neural_network import *

# for plotting results
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.mplot3d import Axes3D                 # for 3d plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
creates a heat map of the test accuracy, depending on number of hidden layers
and nodes per hidden layer
"""

def test_accuracy(data, activation_funcs, layer_output_sizes, 
                  input_size, output_size, cost_fnc, optimizer_):
    # load data
    x_train, x_test, y_train, y_test = data
    activation_derivatives = get_activation_ders(activation_funcs)
    cost_der = globals()[cost_fnc.__name__ + '_der']
    nn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der
        )
    # train the network
    nn.train_network(x_train, y_train, batches=10, optimizer=optimizer_, 
                     epochs=500)
    predicts = nn.predict(x_test)
    return cost_fnc(predicts, y_test)

def test_different_layers(data, number_hidden_layers, nodes_per_layer,
                           input_size, output_size, cost_fnc, optimizer_):
    accuracies = []
    for i in number_hidden_layers:
        accuracy_i = []
        activation_funcs = [sigmoid for _ in range(i+1)]
        for j in nodes_per_layer:
            layer_output_sizes = [j for _ in range(i)]
            layer_output_sizes.append(output_size)
            acc = test_accuracy(data, activation_funcs, layer_output_sizes, 
                                input_size, output_size, cost_fnc, optimizer_)
            accuracy_i.append(acc)
        accuracies.append(accuracy_i)
    return accuracies

def heat_map_test_accuracy(data, number_hidden_layers, nodes_per_layer,
                           input_size, output_size, cost_fnc, optimizer_):
    values = test_different_layers(data, number_hidden_layers, nodes_per_layer,
                               input_size, output_size, cost_fnc, optimizer_)
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


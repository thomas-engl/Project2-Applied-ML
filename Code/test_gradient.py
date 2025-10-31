#run pytest -v in the terminal to run the test

## import other modules
import autograd.numpy as np
import numpy as onp
import sys
import os
sys.path.insert(0, os.getcwd())

## import the python scripts
from neural_network import *
from data_sets import *
from optimizers import *
#from visualize import *
""" load data set """
np.random.seed(248)
n = 1000                                        # choose n = 1000 data points
x_train, x_test, y_train, y_test = load_runge_data(n)

""" initialize the neural network """
_, input_size = np.shape(x_train)
_, output_size = np.shape(y_train)
layer_output_sizes = [5, 10, output_size]     # define number of nodes in layers
activation_funcs = [ReLU, sigmoid, identity]    # activation functions
activation_derivatives = get_activation_ders(activation_funcs)
cost_fnc = mse                                  # cost function
cost_der = mse_der
# construct the network
ffnn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der
        )

gradients_manual=ffnn.compute_gradient(x_train, y_train)

gradients_autograd=ffnn.compute_gradient(x_train, y_train)

def test_gradients_equal():
    for (layer_grad_autograd, layer_grad_manual) in zip(gradients_autograd, gradients_manual):
        (dC_dW_autograd, dC_db_autograd) = layer_grad_autograd
        (dC_dW_manual, dC_db_manual) = layer_grad_manual
        onp.testing.assert_allclose(dC_dW_autograd, dC_dW_manual)
        onp.testing.assert_allclose(dC_db_autograd, dC_db_manual)
            
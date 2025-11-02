# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 00:18:03 2025

@author: HP
"""

### test file for the notebook

## import the python scripts
from neural_network import *
from data_sets import *
from optimizers import *
from compute import *

## import other modules
import autograd.numpy as np
import time

n = 1000                                            # number of points
"""
layer_output_sizes = [100, 100, 100, 100, 100, 
                      100, 1]                       # define number of nodes in layers (output size is 1)
activation_funcs = [ReLU, ReLU, ReLU, LeakyReLU, 
                    LeakyReLU, sigmoid, identity]   # activation functions
"""   
cost_fnc = mse                                      # choose mse as cost function
optimize_algorithm = Momentum(eta=0.001, 
                              momentum=0.9)         # optimizer: gradient descent with momentum
                              
output_size = 1

### 1D Runge function
np.random.seed(123)                                 # set a seed
input_size = 1
data = load_runge_data(n)
"""
mse_first_guess, mse_after_training, _, predictions = test_accuracy(
                                                        data, activation_funcs, layer_output_sizes, 
                                                        input_size, output_size, cost_fnc, optimize_algorithm,
                                                        lambda_=0.015, reg='L2',
                                                        epochs_=500, return_predicts=True)
print("MSE first guess:     ", mse_first_guess)
print("MSE after training:  ", mse_after_training)

### tune hyper parameter lambda for regularization
lmb_L1 = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                            input_size, output_size, cost_fnc, optimize_algorithm, 'L1')
lmb_L2 = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
                            input_size, output_size, cost_fnc, optimize_algorithm, 'L2')

print("optimal parameter for L1 regularization: ", lmb_L1)
print("optimal parameter for L2 regularization: ", lmb_L2)

lambdas_opt = reg_parameters_network_depth(data, input_size, output_size, cost_fnc, 
                                           optimize_algorithm, 'L2')
print(lambdas_opt)

"""
number_hidden_layers = [0, 1, 2, 3, 4, 5]
nodes_per_layer = [10, 20, 30, 40, 50, 60]
# heat map without regularization
start = time.perf_counter()
heat_map_test_accuracy(data, number_hidden_layers, nodes_per_layer,
                       input_size, output_size, cost_fnc, optimize_algorithm,
                       epochs=500)
intermediate = time.perf_counter()
print(intermediate - start)
# heat map with regularization
heat_map_test_accuracy(data, number_hidden_layers, nodes_per_layer,
                       input_size, output_size, cost_fnc, optimize_algorithm,
                       reg = 'L1', epochs=500)
end = time.perf_counter()
print(end - start)

"""
plot_train_test_errs(data, input_size, output_size, cost_fnc, 
                     optimize_algorithm, reg='L2')



#lmb = optimal_reg_parameter(data, activation_funcs, layer_output_sizes, 
#                            input_size, output_size, cost_fnc, 
#                            optimize_algorithm, 'L2', epochs_=1000)
#print("lambda = ", lmb)
test_errs = 0
train_errs = 0
for j in range(10):
    print(j)
    trained_network, _ = init_train_network(
                            data, activation_funcs, layer_output_sizes, 
                            input_size, output_size, cost_fnc, 
                            optimize_algorithm, epochs_=1000)
    test_errs += compute_test_accuracy(trained_network, data)
    train_errs += compute_train_accuracy(trained_network, data)
print("Test error =  ", test_errs / 10)
print("Train error = ", train_errs / 10)
"""

#test number of hidden layers with regularization
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from neural_network import *
from optimizers import *




def tune_hidden_layers_reg(x_train, y_train, x_test, y_test, list_layer_output_sizes, list_activation_funcs, list_number_hidden_layers, list_regularization, epochs):
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    
    results = pd.DataFrame(index=list_number_hidden_layers, columns=list_regularization, dtype=float)
    times = pd.DataFrame(index=list_number_hidden_layers, columns=list_regularization, dtype=float)
    for i, (layer_output_sizes, activation_funcs) in enumerate(zip(list_layer_output_sizes, list_activation_funcs)):
        for lmbda in list_regularization:
            activation_derivatives = get_activation_ders(activation_funcs)
            start = time.perf_counter()
            ffnn = NeuralNetwork(
                input_size, layer_output_sizes, activation_funcs, 
                activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
                )

            ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(0.03), epochs =epochs, printer=False)
            
            end = time.perf_counter()
            results.loc[i+1, lmbda] = accuracy(ffnn.predict(x_test), y_test)
            times.loc[i+1, lmbda] = end-start
            #print("accuraccy of " +str(i) +"is" + str(results.loc[i+1, lmbda]))
            #print(f"took {end - start:.4f} seconds")
    print("--------------------------------------------------")
    print("times:")
    print(times)
    print("accuraccy")
    print(results)
    print("--------------------------------------------------")
    return results

def tune_number_nodes_reg(x_train, y_train, x_test, y_test, list_layer_output_sizes, activation_funcs, list_highest_size, list_regularization, epochs):
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    
    
    results = pd.DataFrame(index=list_highest_size, columns=list_regularization, dtype=float)
    times = pd.DataFrame(index=list_highest_size, columns=list_regularization, dtype=float)
    for i, layer_output_sizes in enumerate(list_layer_output_sizes):
        for lmbda in list_regularization:
            activation_derivatives = get_activation_ders(activation_funcs)
            start = time.perf_counter()
            ffnn = NeuralNetwork(
                input_size, layer_output_sizes, activation_funcs, 
                activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
                )

            ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(0.03), epochs =epochs, printer=False)
            
            end = time.perf_counter()
            results.loc[list_highest_size[i], lmbda] = accuracy(ffnn.predict(x_test), y_test)
            times.loc[list_highest_size[i], lmbda] = end-start
            print("accuraccy of " +str(i) +"is" + str(results.loc[list_highest_size[i], lmbda]))
            print(f"took {end - start:.4f} seconds")
    print("--------------------------------------------------")
    print("times:")
    print(times)
    print("accuraccy")
    print(results)
    print("--------------------------------------------------")    

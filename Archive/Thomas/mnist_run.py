import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from neural_network import *
from optimizers import *
from mnist_functions import *
import sys

import warnings
warnings.filterwarnings("ignore")

with open("output.txt", "w") as f:
    sys.stdout = f

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    # Extract data (features) and target (labels)
    X = mnist.data
    y = mnist.target

    #one hot encoding
    encoder = OneHotEncoder(sparse_output=False)

    # 4. Fit and transform the labels
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    # Scaling
    X = X/255.0

    output_size = 10
    # train test split

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_onehot, test_size=10000, random_state=42
    )

    # Split train+val into train (50k) and validation (10k)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=10000, random_state=42
    )

    #tune number of hidden layers with regularization
    """
    print("tuning of number of hidden layers with regularization")
    np.random.seed(123)
    list_layer_output_sizes = [[128, output_size], [128, 64, output_size], [128, 64, 32, output_size], [128, 64, 32, 16, output_size]]   # define number of nodes in layers
    list_activation_funcs = [[ReLU, softmax], [ReLU, sigmoid, softmax], [ReLU, sigmoid, sigmoid, softmax], [ReLU, sigmoid, sigmoid, sigmoid, softmax]]    # activation functions
    number_hidden_layers =[1,2,3,4]
    list_regularization = np.logspace(-4,1,7)
    list_regularization = np.insert(list_regularization,0,0)




    results_hidden_layer_reg = tune_hidden_layers_reg(X_train, y_train, X_val, y_val, list_layer_output_sizes, list_activation_funcs, number_hidden_layers, list_regularization, epochs=100)
    print("----------------------------------------------------------------------------------")
    
    #tune number of hidden nodes with regularization
    print("tuning of number of hidden notes with regularization")
    np.random.seed(123)
    list_layer_output_sizes = [[512, 256, output_size], [256, 128, output_size], [128, 64, output_size], [64, 32, output_size], [32, 16, output_size]]   # define number of nodes in layers
    activation_funcs = [ReLU, sigmoid, softmax]
    list_highest_size=[512, 256, 128, 64, 32]
    list_regularization = np.logspace(-4,1,7)
    list_regularization = np.insert(list_regularization,0,0)

    results_nodes_reg = tune_number_nodes_reg(X_train, y_train, X_val, y_val, list_layer_output_sizes, activation_funcs, list_highest_size, list_regularization, epochs=100)
    print("---------------------------------------------")
    """

    """
    #fit the best model
    np.random.seed(193)
    _, input_size = np.shape(X_trainval)
    _, output_size = np.shape(y_trainval)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    epochs = 100
    list_numbers = [1,2,3]
    list_regularization = np.logspace(-3,-1, 7)
    list_layer_output_sizes = [[256, 128, 64, output_size], [128, 64, 32, output_size],[256, 128, output_size]]
    list_activation_funcs = [[ReLU, sigmoid, sigmoid, softmax], [ReLU, sigmoid, sigmoid, softmax], [ReLU, sigmoid, softmax]]
    results = pd.DataFrame(index=list_numbers, columns=list_regularization, dtype=float)
    times = pd.DataFrame(index=list_numbers, columns=list_regularization, dtype=float)
    for i, (layer_output_sizes, activation_funcs) in enumerate(zip(list_layer_output_sizes, list_activation_funcs)):
        for lmbda in list_regularization:
            activation_derivatives = get_activation_ders(activation_funcs)
            start = time.perf_counter()
            ffnn = NeuralNetwork(
                input_size, layer_output_sizes, activation_funcs, 
                activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
                )

            ffnn.train_network(X_trainval, y_trainval, batches =100, optimizer=Adam(0.03), epochs =epochs, printer=False)
            
            end = time.perf_counter()
            results.loc[i+1, lmbda] = accuracy(ffnn.predict(X_test), y_test)
            times.loc[i+1, lmbda] = end-start
            #print("accuraccy of " +str(i) +"is" + str(results.loc[i+1, lmbda]))
            #print(f"took {end - start:.4f} seconds")
    print("--------------------------------------------------")
    print("times:")
    print(times)
    print("accuraccy")
    print(results)
    print("--------------------------------------------------")

    np.random.seed(378)
    _, input_size = np.shape(X_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    epochs = 100
    list_numbers = [1,2,3]
    list_regularization = np.logspace(-3,-1, 7)
    list_layer_output_sizes = [[256, 128, 64, output_size], [128, 64, 32, output_size],[256, 128, output_size]]
    list_activation_funcs = [[ReLU, sigmoid, sigmoid, softmax], [ReLU, sigmoid, sigmoid, softmax], [ReLU, sigmoid, softmax]]
    results = pd.DataFrame(index=list_numbers, columns=list_regularization, dtype=float)
    times = pd.DataFrame(index=list_numbers, columns=list_regularization, dtype=float)
    for i, (layer_output_sizes, activation_funcs) in enumerate(zip(list_layer_output_sizes, list_activation_funcs)):
        for lmbda in list_regularization:
            activation_derivatives = get_activation_ders(activation_funcs)
            start = time.perf_counter()
            ffnn = NeuralNetwork(
                input_size, layer_output_sizes, activation_funcs, 
                activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
                )

            ffnn.train_network(X_train, y_train, batches =100, optimizer=Adam(0.03), epochs =epochs, printer=False)
            
            end = time.perf_counter()
            results.loc[i+1, lmbda] = accuracy(ffnn.predict(X_val), y_val)
            times.loc[i+1, lmbda] = end-start
            #print("accuraccy of " +str(i) +"is" + str(results.loc[i+1, lmbda]))
            #print(f"took {end - start:.4f} seconds")
    print("--------------------------------------------------")
    print("times:")
    print(times)
    print("accuraccy")
    print(results)
    print("--------------------------------------------------")
    """

    #train and evaluate best model
    """
    np.random.seed(329)
    layer_output_sizes = [128, 64, 32, output_size]
    activation_funcs = [ReLU, sigmoid, sigmoid, softmax]
    lmbda = 0.021544
    train_and_evaluate_best_model(X_trainval, y_trainval, X_test, y_test, layer_output_sizes, activation_funcs, lmbda)
    print('here')
    """


    #plot errors per epoch
    """"
    np.random.seed(120)
    layer_output_sizes = [128, 64, output_size]
    activation_funcs = [ReLU, sigmoid, softmax]
    #lmbda = 0.021544
    lmbda = 0
    epochs= 200
    compute_errors_per_epoch(X_train, y_train, X_val, y_val, layer_output_sizes, activation_funcs, lmbda, epochs)
    """


    #tune learning rate
    """
    np.random.seed(208)
    layer_output_sizes = [128, 64, 32, output_size]
    activation_funcs = [ReLU, sigmoid, sigmoid, softmax]
    list_lambda = np.logspace(-4,1,5)
    tune_learning_rate(X_train, y_train, X_val, y_val, layer_output_sizes, activation_funcs, list_learning_rates)
    """

    #tune learning rate and lambda simultaneously
    
    np.random.seed(472)
    layer_output_sizes = [128, 64, 32, output_size]
    activation_funcs = [ReLU, sigmoid, sigmoid, softmax]
    list_lambda = np.logspace(-4,1,5)
    list_learning_rates = np.logspace(-4,1,5) 
    tune_learning_rate_reg(X_train, y_train, X_val, y_val, layer_output_sizes, activation_funcs, list_learning_rates, list_lambda, epochs=1)
    
    

    """
    list_layer_output_sizes = [[512, 256, output_size], [256, 128, output_size], [128, 64, output_size], [64, 32, output_size], [32, 16, output_size]]   # define number of nodes in layers
    activation_funcs = [ReLU, sigmoid, softmax]
    list_lambda = np.logspace(-4,1,5)
    list_learning_rates = np.logspace(-4,1,5)
    for layer_output_sizes in list_layer_output_sizes:
        print("---------------------------------")
        print(layer_output_sizes)
        np.random.seed(472)
        tune_learning_rate_reg(X_train, y_train, X_val, y_val, layer_output_sizes, activation_funcs, list_learning_rates, list_lambda, epochs=1)
    """


    sys.stdout = sys.__stdout__ 

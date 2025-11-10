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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm


def compare_activation_reg(x_train, y_train, x_test, y_test, layer_output_sizes, list_activation_funcs, list_model_numbers, list_regularization, epochs):
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    
    results = pd.DataFrame(index=list_model_numbers, columns=list_regularization, dtype=float)
    times = pd.DataFrame(index=list_model_numbers, columns=list_regularization, dtype=float)
    for i, activation_funcs in enumerate(list_activation_funcs):
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
    print("accuraccy of best learning rate")
    print(results.max(axis=1))
    return results

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
    print("accuraccy of best learning rate")
    print(results.max(axis=1))
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
            #print("accuraccy of " +str(i) +"is" + str(results.loc[list_highest_size[i], lmbda]))
            #print(f"took {end - start:.4f} seconds")
    print("--------------------------------------------------")
    print("times:")
    print(times)
    print("accuraccy")
    print(results)
    print("--------------------------------------------------")   
    print("accuraccy of best learning rate")
    print(results.max(axis=1)) 

def train_and_evaluate_best_model(x_train, y_train, x_test, y_test, layer_output_sizes, activation_funcs, lmbda, epochs):
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    activation_derivatives = get_activation_ders(activation_funcs)
    ffnn = NeuralNetwork(
            input_size, layer_output_sizes, activation_funcs, 
            activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
            )

    ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(0.03), epochs =epochs, printer=False)
    print("accuraccy " + str(accuracy(ffnn.predict(x_test), y_test)))
    # Get predicted probabilities
    y_pred = ffnn.predict(x_test)

    # Convert one-hot to integer class indices
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', values_format='d', ax=ax)

    # Get the colorbar from the image in the display
    cbar = disp.im_.colorbar
    cbar.set_label("Number of occurrences", rotation=270, labelpad=15)
    #disp.plot(cmap='Blues', values_format='d')
    #plt.title("Confusion Matrix")
    plt.show()
    return ffnn

def compute_errors_per_epoch(x_train, y_train, x_test, y_test, layer_output_sizes, activation_funcs, lmbda, epochs):
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    activation_derivatives = get_activation_ders(activation_funcs)
    ffnn = NeuralNetwork(
            input_size, layer_output_sizes, activation_funcs, 
            activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
            )

    ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(0.03), epochs =epochs, printer=False, save_mses =True, x_test=x_test, y_test =y_test)
    #print(ffnn.train_errors)
    #print(ffnn.test_errors)

    epoch_list = np.arange(1,(epochs + 1))
    
    plt.figure(dpi=150)
    plt.grid()
    plt.plot(epoch_list, ffnn.train_errors, label = "Training Accuracy")
    plt.plot(epoch_list, ffnn.test_errors, label = "Test Accuracy")
    plt.xlabel("number of epochs")
    plt.legend()
    plt.show()
    

def tune_learning_rate(x_train, y_train, x_test, y_test, layer_output_sizes, activation_funcs, list_learning_rates):
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    layer_output_sizes = [128, 64, 32, output_size]
    activation_funcs = [ReLU, sigmoid, sigmoid, softmax]

    results = [None] * len(list_learning_rates)
    for i, learning_rate in enumerate(list_learning_rates):
        activation_derivatives = get_activation_ders(activation_funcs)
        start = time.perf_counter()
        ffnn = NeuralNetwork(
            input_size, layer_output_sizes, activation_funcs, 
            activation_derivatives, cost_fnc, cost_der,0, 'L2'
            )

        ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(learning_rate), epochs = 100)
        results[i] = accuracy(ffnn.predict(x_test), y_test)
        end = time.perf_counter()

        #print("accuraccy"+ str(results[i]))
        #print(f"took {end - start:.4f} seconds")
    print(results)
    return results

def tune_learning_rate_reg(x_train, y_train, x_test, y_test, layer_output_sizes, activation_funcs, list_learning_rates, list_regularization, epochs):
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
    layer_output_sizes = [128, 64, 32, output_size]
    activation_funcs = [ReLU, sigmoid, sigmoid, softmax]

    results = pd.DataFrame(index=list_learning_rates, columns=list_regularization, dtype=float)
    times = pd.DataFrame(index=list_learning_rates, columns=list_regularization, dtype=float)
    for learning_rate in list_learning_rates:
        for lmbda in list_regularization:
            activation_derivatives = get_activation_ders(activation_funcs)
            start = time.perf_counter()
            ffnn = NeuralNetwork(
                input_size, layer_output_sizes, activation_funcs, 
                activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
                )

            ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(learning_rate), epochs = epochs)
            results.loc[learning_rate, lmbda] = accuracy(ffnn.predict(x_test), y_test)
            end = time.perf_counter()
            times.loc[learning_rate, lmbda] = end-start

        #print("accuraccy"+ str(results[i]))
        #print(f"took {end - start:.4f} seconds")
    print("time")
    print(times)
    print("result")
    print(results)
    z = results.values
    #reg_params = results.columns.to_list()
    #learning_rates = df.index.to_list()
    l, k = len(list_regularization), len(list_learning_rates)

    # plot heatmap
    fig, ax = plt.subplots(dpi=200)
    im = ax.imshow(z, cmap=cm.jet, origin='lower')  # origin='lower' so smaller learning rates are at the bottom

    # ticks and labels
    ax.set_xticks(range(l), labels=[f"{v:.4f}" for v in list_regularization])
    ax.set_yticks(range(k), labels=[f"{v:.4f}" for v in list_learning_rates])
    ax.set_xlabel('regularization parameter')
    ax.set_ylabel('learning rate')


    # colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Test Accuracy", rotation=270, labelpad=15)

    fig.tight_layout()
    plt.show()
    return results


def compare_best_models(X_train, y_train, X_test, y_test, list_layer_output_sizes, list_activation_funcs, list_numbers, list_regularization, epochs):
    _, input_size = np.shape(X_train)
    _, output_size = np.shape(y_train)
    cost_fnc = cross_entropy                                  # cost function
    cost_der = cross_entropy_der
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
    print("accuraccy of best learning rate")
    print(results.max(axis=1)) 





            

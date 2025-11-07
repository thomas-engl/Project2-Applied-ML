import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
#from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.datasets import mnist
from neural_network import *
#from data_sets import *
from optimizers import *
#from visualize import *

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
"""
# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Combine to get the full dataset
import numpy as np
x_full = np.concatenate((x_train, x_test), axis=0)
y_full = np.concatenate((y_train, y_test), axis=0)
plt.imshow(x_full[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()
n_inputs = len(x_full)
x_full = x_full.reshape(n_inputs, -1)

#use only first n samples
n=70000

X_small = x_full[:n]
y_small = y_full[:n]

#one hot encoding
y_encoded = to_categorical(y_small, num_classes=10)

# Scaling
X_small = X_small/255.0

X_small= X_small.reshape(n,-1)
"""
# train test split
x_train, x_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
np.random.seed(348)
#test number of hidden layers
""""
_, input_size = np.shape(x_train)
_, output_size = np.shape(y_train)
cost_fnc = cross_entropy                                  # cost function
cost_der = cross_entropy_der
List_layer_output_sizes = [[128, output_size], [128, 64, output_size], [128, 64, 32, output_size], [128, 64, 32, 16, output_size]]   # define number of nodes in layers
list_activation_funcs = [[ReLU, softmax], [ReLU, sigmoid, softmax], [ReLU, sigmoid, sigmoid, softmax], [ReLU, sigmoid, sigmoid, sigmoid, softmax]]    # activation functions
#List_layer_output_sizes = [[128, 64, output_size]]# [128, 64, 32, 16, output_size]]   # define number of nodes in layers
#list_activation_funcs = [[ReLU, ReLU, softmax]] #[ReLU, sigmoid, ReLU, sigmoid, softmax]]    # activation functions
results = [None] * len(list_activation_funcs)
times = [None] * len(list_activation_funcs)
for i, (layer_output_sizes, activation_funcs) in enumerate(zip(List_layer_output_sizes, list_activation_funcs)):
    activation_derivatives = get_activation_ders(activation_funcs)
    start = time.perf_counter()
    ffnn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der,0, 'L1'
        )

    ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(0.005), epochs =200, printer=True, x_test= x_test, y_test =y_test)
    
    end = time.perf_counter()
    results[i] = accuracy(ffnn.predict(x_test), y_test)
    times[i] = end-start
    print("accuraccy of " +str(i) +"is" + str(results[i]))
    print(f"took {end - start:.4f} seconds")
print("--------------------------------------------------")
print("times:")
print(times)
print("accuraccy")
print(results)
print("--------------------------------------------------")

"""
#test number of hidden layers with regularization
_, input_size = np.shape(x_train)
_, output_size = np.shape(y_train)
cost_fnc = cross_entropy                                  # cost function
cost_der = cross_entropy_der
List_layer_output_sizes = [[128, output_size], [128, 64, output_size], [128, 64, 32, output_size], [128, 64, 32, 16, output_size]]   # define number of nodes in layers
list_activation_funcs = [[ReLU, softmax], [ReLU, sigmoid, softmax], [ReLU, sigmoid, sigmoid, softmax], [ReLU, sigmoid, sigmoid, sigmoid, softmax]]    # activation functions

number_hidden_layers =[1,2,3,4]
List_regularization = np.logspace(-4,1,5)
#List_layer_output_sizes = [[128, 64, output_size]]# [128, 64, 32, 16, output_size]]   # define number of nodes in layers
#list_activation_funcs = [[ReLU, ReLU, softmax]] #[ReLU, sigmoid, ReLU, sigmoid, softmax]]    # activation functions
results = pd.DataFrame(index=number_hidden_layers, columns=List_regularization, dtype=float)
times = pd.DataFrame(index=number_hidden_layers, columns=List_regularization, dtype=float)
for i, (layer_output_sizes, activation_funcs) in enumerate(zip(List_layer_output_sizes, list_activation_funcs)):
    for lmbda in List_regularization:
        activation_derivatives = get_activation_ders(activation_funcs)
        start = time.perf_counter()
        ffnn = NeuralNetwork(
            input_size, layer_output_sizes, activation_funcs, 
            activation_derivatives, cost_fnc, cost_der, lmbda, 'L2'
            )

        ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(0.03), epochs =100, printer=True)
        
        end = time.perf_counter()
        results.loc[i+1, lmbda] = accuracy(ffnn.predict(x_test), y_test)
        times.loc[i+1, lmbda] = end-start
        print("accuraccy of " +str(i) +"is" + str(results.loc[i+1, lmbda]))
        print(f"took {end - start:.4f} seconds")
print("--------------------------------------------------")
print("times:")
print(times)
print("accuraccy")
print(results)
print("--------------------------------------------------")


"""
#test learning rate
_, input_size = np.shape(x_train)
_, output_size = np.shape(y_train)
cost_fnc = cross_entropy                                  # cost function
cost_der = cross_entropy_der
List_learning_rates = np.logspace(-5,0,5)  # define number of nodes in layers
layer_output_sizes = [128, 64, 32, output_size]
activation_funcs = [ReLU, sigmoid, sigmoid, softmax]

results = [None] * len(List_learning_rates)
for i, learning_rate in enumerate(List_learning_rates):
    activation_derivatives = get_activation_ders(activation_funcs)
    start = time.perf_counter()
    ffnn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der,0, 'L1'
        )

    ffnn.train_network(x_train, y_train, batches =100, optimizer=RMS_prop(learning_rate), epochs =100)
    results[i] = accuracy(ffnn.predict(x_test), y_test)
    end = time.perf_counter()

    print("accuraccy"+ str(results[i]))
    print(f"took {end - start:.4f} seconds")
"""
#test regularization

""""
_, input_size = np.shape(x_train)
_, output_size = np.shape(y_train)
cost_fnc = cross_entropy                                  # cost function
cost_der = cross_entropy_der
List_regularization = np.logspace(-4,1,5)  # define number of nodes in layers
layer_output_sizes = [128, 64, 32, output_size]
activation_funcs = [ReLU, sigmoid, sigmoid, softmax]

results = [None] * len(List_regularization)
for i, lmbda in enumerate(List_regularization):
    activation_derivatives = get_activation_ders(activation_funcs)
    start = time.perf_counter()
    ffnn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der,lmbda, 'L2'
        )

    ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(3*0.01), epochs =100)
    results[i] = accuracy(ffnn.predict(x_test), y_test)
    end = time.perf_counter()

    print("accuraccy for"+str(lmbda) +":"+ str(results[i]))
    print(f"took {end - start:.4f} seconds")
"""

#test number of hidden nodes
"""
_, input_size = np.shape(x_train)
_, output_size = np.shape(y_train)
cost_fnc = cross_entropy                                  # cost function
cost_der = cross_entropy_der
List_layer_output_sizes = [[512, 256, output_size], [256, 128, output_size], [128, 64, output_size], [64, 32, output_size], [32, 16, output_size]]   # define number of nodes in layers
activation_funcs = [ReLU, sigmoid, softmax]
#List_layer_output_sizes = [[128, 64, output_size]]# [128, 64, 32, 16, output_size]]   # define number of nodes in layers
#list_activation_funcs = [[ReLU, ReLU, softmax]] #[ReLU, sigmoid, ReLU, sigmoid, softmax]]    # activation functions
results = [None] * len(List_layer_output_sizes)
for i, layer_output_sizes in enumerate(List_layer_output_sizes):
    activation_derivatives = get_activation_ders(activation_funcs)
    start = time.perf_counter()
    ffnn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der,0, 'L1'
        )

    ffnn.train_network(x_train, y_train, batches =100, optimizer=Adam(0.003), epochs =100)
    results[i] = accuracy(ffnn.predict(x_test), y_test)
    end = time.perf_counter()

    print("accuraccy"+ str(results[i]))
    print(f"took {end - start:.4f} seconds")
"""
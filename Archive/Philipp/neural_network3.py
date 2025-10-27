# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:31:10 2025

@author: Philipp Brückelt
"""
from copy import deepcopy
# autograd and sklearn
import autograd.numpy as np
from autograd import grad, elementwise_grad
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
# for plotting results
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.mplot3d import Axes3D                 # for 3d plotting
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for interpolating predictions for visualizing
from scipy import interpolate


"""
Activation functions
"""

def ReLU(z):
    return np.where(z > 0, z, 0)

def LeakyReLU(z):
    return np.where(z > 0, z, 0.01 * z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    return np.exp(z) / np.sum(np.exp(z), axis=1)[:, np.newaxis]

def identity(z):
    return z

def ELU(z):
    return np.where(z > 0, z, 0.01 * (np.exp(z) - 1))

"""
derivatives of activation functions
"""

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def LeakyReLU_der(z):
    return np.where(z > 0, 1, 0.01)

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax_der(z):
    pass

def identity_der(z):
    return 1

def ELU_der(z):
    return np.where(z > 0, 1, 0.01 * np.exp(z))

"""
create list of activation derivatives from list of activation functions
"""

def get_activation_ders(activation_funcs):
    ders = []
    for fnc in activation_funcs:
        der_name = fnc.__name__ + '_der'
        ders.append(globals()[der_name])
    return ders

"""
some cost functions
"""

def mse(predict, target):
    return np.mean((predict - target) ** 2)

def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

"""
derivatives of cost functions
"""

def mse_der(predict, target):
    return 2 / predict.size * (predict - target)

def cross_entropy_der(predict, traget):
    pass


"""
accuracy of the model
"""

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)

""" ===========================================================================
                            Optimizers
=========================================================================== """

class Scheduler:
    """
    Abstract class for Schedulers
    """

    def __init__(self, eta):
        self.eta = eta

    # should be overwritten
    def update_change(self, gradient):
        raise NotImplementedError

    # overwritten if needed
    def reset(self):
        pass


class Constant(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)

    def update_change(self, gradient):
        return self.eta * gradient
    
    def reset(self):
        pass


class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float):
        super().__init__(eta)
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change

    def reset(self):
        pass


class Adagrad(Scheduler):
    def __init__(self, eta):
        super().__init__(eta)
        self.G_t = None

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T
        G_t_inverse = 1 / (
            delta + np.sqrt(np.diag(self.G_t))
        )
        if len(gradient.shape) != 1:
            G_t_inverse = G_t_inverse.reshape(-1, 1)
        return self.eta * gradient * G_t_inverse

    def reset(self):
        self.G_t = None


class AdagradMomentum(Scheduler):
    def __init__(self, eta, momentum):
        super().__init__(eta)
        self.G_t = None
        self.momentum = momentum
        self.change = 0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.diag(self.G_t))
        )
        if len(gradient.shape) != 1:
            G_t_inverse = G_t_inverse.reshape(-1, 1)
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self):
        self.G_t = None


class RMS_prop(Scheduler):
    def __init__(self, eta, rho=0.9):
        super().__init__(eta)
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self):
        self.second = 0.0


class Adam(Scheduler):
    def __init__(self, eta, rho=0.9, rho2=0.999):
        super().__init__(eta)
        self.rho = rho
        self.rho2 = rho2
        self.moment = 0
        self.second = 0
        self.n_epochs = 1

    def update_change(self, gradient):
        delta = 1e-8  # avoid division ny zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self):
        self.n_epochs += 1
        self.moment = 0
        self.second = 0
""" ===========================================================================
                            Neural network code
=========================================================================== """

class NeuralNetwork:
    
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        activation_ders,
        cost_fun,
        cost_der,
    ):
        self.i_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.layers = self.create_layers_batch()
        self.a_fncs = activation_funcs
        self.a_ders = activation_ders
        self.cost_fnc = cost_fun
        self.cost_der = cost_der
        
    def create_layers_batch(self):
        layers = []
        i_size = self.i_size
        for layer_output_size in self.layer_output_sizes:
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(layer_output_size)
            layers.append((W, b))
            i_size = layer_output_size
        return layers

    def predict(self, inputs):
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.a_fncs):
            z = a @ W + b.reshape(1, -1)
            a = activation_func(z)
        return a

    def cost(self, inputs, targets):
        predictions = self.predict(self, inputs)
        return self.cost_fnc(predictions, targets)

    def _feed_forward_saver(self, inputs):
        layer_inputs = []
        zs = []
        a = inputs
        for (W, b), activation_func in zip(self.layers, self.a_fncs):
            layer_inputs.append(a)
            z = a @ W + b.reshape(1, -1)
            a = activation_func(z)
            zs.append(z)
        return layer_inputs, zs, a

    def compute_gradient(self, inputs, targets):
        #add regularization term
        layer_inputs, zs, predict = self._feed_forward_saver(inputs)
        layer_grads = [() for layer in self.layers]
        # We loop over the layers, from the last to the first
        for i in reversed(range(len(self.layers))):
            layer_input, z, activation_der = layer_inputs[i], zs[i], self.a_ders[i]
            if i == len(self.layers) - 1:
                if self.cost_der == cross_entropy_der and self.a_fncs[i] == softmax:
                    dC_dz = predict - targets
                # For last layer we use cost derivative as dC_da(L) can be 
                # computed directly
                else:
                    dC_da = self.cost_der(predict, targets)
                    dC_dz = dC_da * activation_der(z)
            else:
                # For other layers we build on previous z derivative, as 
                # dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
                (W, b) = self.layers[i + 1]
                dC_da = dC_dz @ W.T
                dC_dz = dC_da * activation_der (z)
            dC_dW = layer_input.T @ dC_dz
            # we take the sum of all derivatives dC_dz to stay consistent 
            # with autograd
            dC_db = np.sum(dC_dz, axis=0)
            layer_grads[i] = (dC_dW, dC_db)
        return layer_grads
    
    def train_network(self, inputs, targets, batches, optimizer, epochs):
        batch_size = inputs.shape[0] // batches
        optimizers_weight = []
        optimizers_bias = []
        for i in range(len(self.layers)):
            optimizers_weight.append(deepcopy(optimizer))
            optimizers_bias.append(deepcopy(optimizer))
        
        for _ in range(epochs):
            inputs_resampled, targets_resampled = resample(inputs, targets)
            for i in range(batches):
                if i == (batches - 1):
                # If the for loop has reached the last batch, take all thats left
                    inputs_batch = inputs_resampled[i * batch_size :, :]
                    targets_batch = targets_resampled[i * batch_size :, :]
                else:
                    inputs_batch = inputs_resampled[i * batch_size : (i + 1) * batch_size, :]
                    targets_batch = targets_resampled[i * batch_size : (i + 1) * batch_size, :]
            updated_layers = []
            layers_grad = self.compute_gradient(inputs_batch, targets_batch)
            for j in range(len(layers_grad)):
                W, b = self.layers[j]
                W_g, b_g = layers_grad[j]
                W -= optimizers_weight[j].update_change(W_g)
                b -= optimizers_bias[j].update_change(b_g)
                updated_layers.append((W,b))
            self.layers = updated_layers


    """ These last two methods are not needed in the project, but they can be 
    nice to have! The first one has a layers parameter so that you can use 
    autograd on it """
    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        pass
             

        
"""
Scaling data
"""
        
def get_inputs_targets(inputs, targets):
    """
    get correct input and target shapes, in case input or output size is 1
    avoids manual reshaping
    """
    if len(np.shape(inputs)) == 1:
        # if inputs are of the shape (n,)
        x = inputs.reshape(-1, 1)
    else:
        x = inputs
    if len(np.shape(targets)) == 1:
        y = targets.reshape(-1, 1)
    else:
        y = targets
    return x, y

def get_scaled_data(inputs, targets):
    """ scale the data """
    x, y = get_inputs_targets(inputs, targets)
    # scaler = StandardScaler()
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.fit_transform(x_test)
    # y_mean = y_train.mean()
    # y_train -= y_mean
    # y_test -= y_mean
    return x_train, x_test, y_train, y_test


""" ===========================================================================
                        Code for functions to approximate
        sample all the data here so that we only have to load it later
        
We consider the following functions:
    - Runge (1D and 2D)
    - a Gaussian
    - Rastrigin
============================================================================"""

def runge(x): return 1 / (1 + 25*x**2)

def load_runge_data(num_pts):
    x = np.random.uniform(-1, 1, num_pts)               # random data points
    y = runge(x) + np.random.normal(0, 0.1, num_pts)   # targets
    return get_scaled_data(x, y)

def runge2D(x, y): return 1 / (1 + (10*x - 5)**2 + (10*y - 5)**2)

def load_runge2D_data(num_pts):
    x = np.random.uniform(0, 1, size=(num_pts, 2))
    y = runge2D(x[:,0], x[:,1]) + np.random.normal(0, 0.1, num_pts)
    return get_scaled_data(x, y)

def gaussian_2D(x, y): return np.exp(- np.pi * (x**2 + y**2))

def load_gaussian2D_data(num_pts):
    x = np.random.uniform(-1, 1, size=(num_pts, 2))
    y = gaussian_2D(x[:,0], x[:,1]) + np.random.normal(0, 0.1, num_pts)
    return get_scaled_data(x, y)

def rastrigin(x, y): 
    return 20 + x**2 - 10*np.cos(2*np.pi*x) + y**2 - 10*np.cos(2*np.pi*y)

def load_rastrigin_data(num_pts):
    x = np.random.uniform(-1, 1, size=(num_pts, 2))
    y = rastrigin(x[:,0], x[:,1]) + np.random.normal(0, 0.1, num_pts)
    return get_scaled_data(x, y)
    
def plot_3D(f, domain):
    x, y = np.linspace(domain[0][0], domain[0][1]), np.linspace(domain[1][0], 
                                                                domain[1][1])
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    # surface plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(8,5), 
                           dpi=150)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, 
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.set_xlabel('$x$', fontsize=15)
    ax.set_ylabel('$y$', fontsize=15)
    ax.yaxis._axinfo['label']['space_factor'] = 3.0
    # disable auto rotation 
    ax.set_zlabel('$f(x, y)$', fontsize=15, rotation = 0)
    ax.zaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))
    plt.show()
    
def plot_approximation_3D(x_data, y_data, domain):
    # interplate the discrete values
    f = interpolate.LinearNDInterpolator(x_data, y_data.reshape(-1,))
    plot_3D(f, domain)
    

"""============================================================================
                Visualize results of numerical experimenst
============================================================================"""    

"""
creates a heat map of the test accuracy, depending on number of hidden layers
and nodes per hidden layer
"""

def test_accuracy(activation_funcs, layer_output_sizes, cost_fnc=mse):
    activation_derivatives = get_activation_ders(activation_funcs)
    cost_der = globals()[cost_fnc.__name__ + '_der']
    nn = NeuralNetwork(
        input_size, layer_output_sizes, activation_funcs, 
        activation_derivatives, cost_fnc, cost_der
        )
    # train the network
    nn.train_network(x_train, y_train, batches=10, optimizer=Momentum(
        eta=0.01, momentum=0.9), epochs=500)
    predicts = nn.predict(x_test)
    return cost_fnc(predicts, y_test)

def test_different_layers(data, number_hidden_layers, nodes_per_layer,
                           input_size, output_size, cost_fnc=mse):
    # load data
    x_train, x_test, y_train, y_test = data
    accuracies = []
    for i in number_hidden_layers:
        accuracy_i = []
        activation_funcs = [sigmoid for _ in range(i+1)]
        for j in nodes_per_layer:
            layer_output_sizes = [j for _ in range(i)]
            layer_output_sizes.append(output_size)
            acc = test_accuracy(activation_funcs, layer_output_sizes, cost_fnc)
            accuracy_i.append(acc)
        accuracies.append(accuracy_i)
    return accuracies

def heat_map_test_accuracy(data, number_hidden_layers, nodes_per_layer,
                           input_size, output_size, cost_fnc=mse):
    values = test_different_layers(data, number_hidden_layers, nodes_per_layer,
                               input_size, output_size, cost_fnc)
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



if __name__ == '__main__':
    
    np.random.seed(0)
    
    """ load data set """
    n = 1000                                        # choose n = 1000 data points
    x_train, x_test, y_train, y_test = load_runge_data(n)
    
    """ initialize the neural network """
    _, input_size = np.shape(x_train)
    _, output_size = np.shape(y_train)
    layer_output_sizes = [50, 100, output_size]     # define number of nodes in layers
    activation_funcs = [ReLU, sigmoid, identity]    # activation functions
    activation_derivatives = get_activation_ders(activation_funcs)
    cost_fnc = mse                                  # cost function
    cost_der = mse_der
    # construct the network
    ffnn = NeuralNetwork(
            input_size, layer_output_sizes, activation_funcs, 
            activation_derivatives, cost_fnc, cost_der
            )
    
    """ make first predictions on the train data and compute the 
    cost function """
    predicts = ffnn.predict(x_train)
    print(mse(predicts, y_train))
    
    """ train the network and compute new predictions on the test data """
    ffnn.train_network(x_train, y_train, batches= 10, optimizer=Momentum(
        eta=0.01, momentum=0.9), epochs=500)
    predicts_new = ffnn.predict(x_test)
    print(mse(predicts_new, y_test))
    
    """ plot the dependence of the test accuracy on the number of hidden layers
    and number of nodes per layer. Here, we only use the sigmoid function
    as activation function """
    heat_map_test_accuracy(load_runge_data(n), [0, 1, 2, 3], [5, 10, 25, 50], 
                           input_size, output_size)

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 14:31:10 2025

@author: Philipp Brückelt
"""


# autograd and sklearn
import autograd.numpy as np
from autograd import grad, elementwise_grad
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split 
from sklearn.preprocessing import StandardScaler

# for plotting results
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, LogLocator
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import StrMethodFormatter
from mpl_toolkits.mplot3d import Axes3D                 # for 3d plotting


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

    """
    def update_weights(self, layer_grads, velocity):
        # Work on a copy of the layers so the caller's objects are not mutated 
        # unexpectedly
        new_layers = [(W.copy(), b.copy()) for (W, b) in self.layers]
        self.layers = gd_momentum_step(new_layers, layer_grads, velocity)
    """

    """ These last two methods are not needed in the project, but they can be 
    nice to have! The first one has a layers parameter so that you can use 
    autograd on it """
    def autograd_compliant_predict(self, layers, inputs):
        pass

    def autograd_gradient(self, inputs, targets):
        pass
    
    
"""============================================================================
                        Training of the network
============================================================================"""

def get_params(network, optimizer):
    """
    initialize parameters for optimizers
    these are set to 0 for the start, only make sure that the dimensions are
    correct
    """
    params = []
    for i in range(len(network.layers)):
        W, b = network.layers[i]
        W0, b0 = np.zeros_like(W), np.zeros_like(b)
        if optimizer in [gd_momentum, adagrad, RMSProp]:
            params.append((W0, b0))
        elif optimizer == adam:
            params.append((W0, W0, b0, b0))
    return params
        
def train_network(network, inputs, targets, optimizer, epochs=100, 
                  learning_rate=0.001, momentum=0.9, tol=1e-5):
    if optimizer != gd:
        params = get_params(network, optimizer)
    if optimizer == gd_momentum:
        # if we use gd with momentum, we add the momentum to the parameter list
        params.append(momentum)
    if optimizer == adam:
        # for adam, we also need the current epoch with start from 1,
        # will be update in every step
        params.append(1)
    for epoch in range(epochs):
        gradients = network.compute_gradient(inputs, targets)
        if optimizer == gd:
            network.layers = gd(network.layers, gradients)
        else:
            network.layers, params = optimizer(network.layers, gradients,
                                               params, learning_rate)
        # check stopping criterion (only at every 10th epoch to avoid too many
        # function calls)
        if epoch % 10 == 0:
            predicts = ffnn.predict(inputs)
            # break if cost function is small enough
            if ffnn.cost_fnc(predicts, targets) < tol:
                break
            
            
"""
Gradient descent methods
performs only a single step
"""

def gd(layers, gradients, learning_rate=0.001):  
    new_layers = []
    for i in range(len(layers)):
        # update W
        W = layers[i][0] - learning_rate * gradients[i][0]
        b = layers[i][1] - learning_rate * gradients[i][1]
        new_layers.append((W, b))
    return new_layers

def gd_momentum(layers, gradients, params, learning_rate=0.001):
    # params contains velocity and momentum in the last entry
    updates = []
    # Update each layer using SGD with momentum
    for i in range(len(layers)):
        W, b = layers[i]
        W_g, b_g = gradients[i]
        vW, vb = params[i]
        # velocity update (momentum) and parameter step
        vW = params[-1] * vW + learning_rate * W_g
        vb = params[-1] * vb + learning_rate * b_g
        W -= vW
        b -= vb
        updates.append((W, b))
        params[i] = (vW, vb)
    return updates, params

def adam(layers, gradients, params, learning_rate, epsilon = 1e-7, 
                          beta_1 = 0.9, beta_2 = 0.999):
    new_layers = []
    new_params = []
    epoch = params[-1]
    for i in range(len(layers)):
        m_W, v_W, m_b, v_b = params[i]
        W_g, b_g = gradients[i]
        m_W = beta_1 * m_W + (1-beta_1)*W_g
        v_W = beta_2 * v_W + (1-beta_2)*W_g**2
        m_b = beta_1 * m_b + (1-beta_1)*b_g
        v_b = beta_2 * v_b + (1-beta_2)*b_g**2
        m_W_tilde = m_W/(1-beta_1**epoch)
        v_W_tilde = v_W/(1-beta_2**epoch)
        m_b_tilde = m_b/(1-beta_1**epoch)
        v_b_tilde = v_b/(1-beta_2**epoch)
        W = layers[i][0] - learning_rate * m_W_tilde / (np.sqrt(v_W_tilde
                                                                ) + epsilon)
        b = layers[i][1] - learning_rate * m_b_tilde / (np.sqrt(v_b_tilde
                                                                ) + epsilon)
        new_layers.append((W, b))
        new_params.append((m_W, v_W, m_b, v_b))
    # update the last parameter
    epoch += 1
    new_params.append(epoch)
    return new_layers, new_params

def adagrad(layers, gradients, G_mats, learning_rate, epsilon = 1e-7):
    # needs a much higher learning rate, e.g., 0.1
    new_layers, new_mats = [], []
    for i in range(len(layers)):
        W, b = layers[i]
        g_W, g_b = gradients[i]
        G_W, G_b = G_mats[i]
        G_W = G_W + g_W**2
        G_b = G_b + g_b**2
        W_new = W - learning_rate * g_W / np.sqrt(epsilon + G_W)
        b_new = b - learning_rate * g_b / np.sqrt(epsilon + G_b)
        new_mats.append((G_W, G_b))
        new_layers.append((W_new, b_new))
    return new_layers, new_mats

def RMSProp(layers, gradients, params, learning_rate, epsilon=1e-7, rho=0.9):
    # a good learning rate is approximately 0.001
    new_layers, new_params = [], []
    for i in range(len(layers)):
        v_W, v_b = params[i]
        g_W, g_b = gradients[i]
        W, b = layers[i]
        v_W = rho * v_W + (1 - rho) * g_W**2
        v_b = rho * v_b + (1 - rho) * g_b**2
        W_new = W - learning_rate / np.sqrt(v_W + epsilon) * g_W
        b_new = b - learning_rate / np.sqrt(v_b + epsilon) * g_b
        new_layers.append((W_new, b_new))
        new_params.append((v_W, v_b))
    return new_layers, new_params
        
        
        
        
   
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
    train_network(ffnn, x_train, y_train, optimizer=adam, epochs=500, 
                  learning_rate=0.01)
    predicts_new = ffnn.predict(x_test)
    print(mse(predicts_new, y_test))
    
    plt.scatter(x_test, y_test, color='blue')
    plt.scatter(x_test, predicts_new, color='red')
    
    """
    domain = [[0, 1], [0, 1]]
    plot_approximation_3D(x_test, y_test, domain)             # true solution
    plot_approximation_3D(x_test, predicts_new, domain)       # approximation
    """

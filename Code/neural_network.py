# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:27:00 2025

@author: Lars Bosch, Philipp Brückelt, Thomas Engl
"""


from copy import deepcopy
# autograd and sklearn
import autograd.numpy as np
from autograd import grad, elementwise_grad
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


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
        lmd=0.0,
        regularization=None
    ):
        """
        initializes the neural network and sets up the parameters

        Parameters
        ----------
        network_input_size : int
            input size of the network
        layer_output_sizes : list
            list with output sizes of the hidden layers and the output layer
        activation_funcs : list
            list of activation functions, i-th entry correspons to the i-th layer
        activation_ders : list
            list of derivatives of activation functions
        cost_fun : function
            cost function
        cost_der : function
            derivative of the cost function
        lmd : float, optional
            regularization parameter. The default is 0.0.
        regularization : string, optional
            type of regularization, either 'L1', 'L2' or None. The default is None.

        Returns
        -------
        None.

        """
        self.i_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.layers = self.create_layers_batch()
        self.a_fncs = activation_funcs
        self.a_ders = activation_ders
        self.cost_fnc = cost_fun
        self.cost_der = cost_der
        self.lmd = lmd
        self.regularization = regularization
        
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
            # if regularization is used, add the derivative w.r.t to the weights
            if self.regularization == 'L2' and self.lmd > 0.0:
                dC_dW += 2 * self.lmd * W
            elif self.regularization == 'L1' and self.lmd > 0.0:
                dC_dW += self.lmd * np.sign(W)
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
        a = inputs
        for (W, b), activation_func in zip(layers, self.a_fncs):
            z = a @ W + b.reshape(1, -1)
            a = activation_func(z)
        return a

    def autograd_gradient(self, inputs, targets):
        # first define the cost function we are taking the derivatives of
        def cost(input_, layers, target):
            predictions = self.autograd_compliant_predict(layers, input_)
            return self.cost_fnc(predictions, target)
        # gradient wrt layers
        grad_func = grad(cost, 1)
        return grad_func(inputs, self.layers, targets)

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:32:13 2025

@author: Lars Bosch, Philipp Brückelt, Thomas Engl
"""


# autograd and sklearn
import autograd.numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.datasets import fetch_openml
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

def get_scaled_data(x_train, x_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    y_mean = y_train.mean()
    y_train -= y_mean
    y_test -= y_mean
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

def load_runge_data(num_pts, sigma=0.1):
    x = np.random.uniform(-1, 1, num_pts)                # random data points
    y = runge(x) + np.random.normal(0, sigma, num_pts)   # targets
    x, y = get_inputs_targets(x, y)
    return train_test_split(x, y)

def runge2D(x, y): return 1 / (1 + (10*x - 5)**2 + (10*y - 5)**2)

def load_runge2D_data(num_pts, sigma=0.1):
    x = np.random.uniform(0, 1, size=(num_pts, 2))
    y = runge2D(x[:,0], x[:,1]) + np.random.normal(0, sigma, num_pts)
    x, y = get_inputs_targets(x, y)
    return train_test_split(x, y)

def gaussian_2D(x, y): return np.exp(- np.pi * (x**2 + y**2))

def load_gaussian2D_data(num_pts):
    x = np.random.uniform(-1, 1, size=(num_pts, 2))
    y = gaussian_2D(x[:,0], x[:,1]) + np.random.normal(0, 0.1, num_pts)
    x, y = get_inputs_targets(x, y)
    return train_test_split(x, y)

""" Rastrigin function for arbitrary dimensions d >= 2 """

def rastrigin(x):
    x = np.asarray(x)
    d = x.shape[-1]
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=-1)

def load_rastrigin_data(num_pts, d, sigma=0.1):
    """
    Erzeugt Trainingsdaten (x, y) für die d-dimensionale Rastrigin-Funktion.
    """
    x = np.random.uniform(-1, 1, size=(num_pts, d))
    y = rastrigin(x) + np.random.normal(0, sigma, num_pts)
    x, y = get_inputs_targets(x, y)
    return train_test_split(x, y)
    
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


""" MNIST data set """

def load_mnist():
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    # Extract data (features) and target (labels)
    x = mnist.data
    y = mnist.target
    # scale and train-test split
    x /= 255.0
    y /= 255.0
    return train_test_split(x, y)

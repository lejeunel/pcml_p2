import pdb
import numpy as np
import matplotlib.pyplot as plt

def plot_training_progress(e_train, e_val):
    """
    Tis functions plots the training and the validation error over all
    iterations of the fitting algorithm
    INPUTS
    @e_train: array of size (nb_iter x 1), containing the training error for each of the nb_iter iterations
    @e_val: array of size (nb_iter x 1), containing the validation error for each of the nb_iter iterations
    """
    train = plt.plot(e_train, label = 'Training error')
    plt.hold(True)
    val = plt.plot(e_val, label = 'Validation error')
    plt.legend(handles = [train, val])
    plt.show()

def plot_cross_validation():

def plot_methods_comparison():

""" example decision boundary functions """
import numpy as np
from numbers import Number

def flat(x: Number):
    """ flat line """
    return 0.8 * x**0

def diag(x: Number):
    """ diagonal straight line (y=x: Number)"""
    return x

def diag_sine(x: Number):
    """ diagonal sine wave """
    return 0.08*np.sin(x*7*np.pi)+x

def arch(x: Number):
    return -(x-0.5)**2 + 0.75

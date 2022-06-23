import numpy as np

from src.distance import Distance

def fig_requirements(func):
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    return func

"""Diagonal straight line, with plot"""
@fig_requirements
def func(x):
    return x
Distance(p=(0.1,0.4), dbf=func, b={'x':[0,1],'y':[0,1]}, norm=True, plot=True).execute()


"""Diagonal sine wave, no plot"""
def func(x):
    return 0.08*np.sin(x*7*np.pi)+x
Distance(p=(0.1,0.02), dbf=func, b={'x':[0,1],'y':[0,1]}, norm=True, plot=False).execute()

"""Diagonal sine wave, with plot"""
@fig_requirements
def func(x):
    return 0.08*np.sin(x*7*np.pi)+x
# Note the bottom right corner is not the max distance edge point as these have a smaller minimum distance to points
# on the boundary to the right of MB
Distance(p=(0.1,0.02), dbf=func, b={'x':[0,1],'y':[0,1]}, norm=True, plot=True).execute()

"""Arch, with plot"""
@fig_requirements
def func(x):
    return -(x-0.5)**2 + 0.75
# Note the bottom right corner is not the max distance edge point as these have a smaller minimum distance to points
# on the boundary to the right of MB
Distance(p=(0.6,0.5), dbf=func, b={'x':[0,1],'y':[0,1]}, norm=True, plot=True).execute()

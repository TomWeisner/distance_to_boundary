""" method to demonstrate the Distance class visually by plotting the results, uses child class DistancePlot """
from src.distancePlot import DistancePlot
from src.utils.datatypes import Bounds
from src.utils.dbfs import diag, diag_sine, arch, flat

bounds = Bounds(bound_dict={'x': [0, 1], 'y': [0, 1]})


a = DistancePlot(p=(0.1,0.4), dbf=flat, b=bounds, norm=True).execute_and_plot()

b = DistancePlot(p=(0.1,0.4), dbf=diag, b=bounds, norm=True).execute_and_plot()

c = DistancePlot(p=(0.1,0.4), dbf=diag_sine, b=bounds, norm=True).execute_and_plot()

d = DistancePlot(p=(0.1,0.4), dbf=arch, b=bounds, norm=True).execute_and_plot()

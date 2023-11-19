from scipy.optimize import minimize_scalar
import numpy as np
from abc import ABC
from typing import Callable, Dict, TypedDict,Tuple

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import copy
from src.utils.datatypes import Point, Bounds, DistanceOutput
from src.distance import Distance

class DistancePlot(Distance):
    """Plot the results of determining the closest point on a decision boundary to a point P """

    def __init__(self, p: Point, dbf: Callable, b: Dict[str, Point], norm=False):
        self.p_copy = copy.copy(p)

        super().__init__(p=p, dbf=dbf, b=b, norm=norm)

    def execute_and_plot(self):
        output: DistanceOutput = self.execute()
        self.__plot()

        return output

    def __plot(self):
        ax = self.__construct_figure()

        if self.norm:
            self.__norm_plot(ax=ax)

    def __construct_figure(self) -> object:
        self.p = self.p_copy # reset the variable if previous calc has overwritten it

        fig, ax = plt.subplots()
        ax.set_xlim((self.bounds.x.lower, self.bounds.x.upper))
        ax.set_ylim((self.bounds.y.lower, self.bounds.y.upper))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        x = np.arange(self.bounds.x.lower, self.bounds.x.upper + 0.01, 0.005)

        ax.plot(self.p[0], self.p[1], 'r.', markersize=10, label=self.__get_point_label('P', self.p), clip_on=False)
        ax.plot(self.boundary_point[0], self.boundary_point[1], 'g.', label=self.__get_point_label('CB', self.boundary_point), markersize=10, clip_on=False)
        ax.plot(x, self._f(x), '-', color='grey', label='Boundary', clip_on=False)
        ax.plot(*self.__get_line(boundary_point=self.boundary_point, p=self.p), 'b--', label=f'Distance = {round(self.distance, 3)}', clip_on=False)
        ax.legend()

        return ax

    def __get_point_label(self, name, point) -> str:
        point = [float(i) if type(i) == np.ndarray else i for i in point] # array to float so can define round method
        x, y = point[0], point[1]
        return f'{name} = ({x:.2f}, {y:.2f})'

    def __norm_plot(self, ax) -> None:
        ax.plot(self.max_distance_edge_point[0], self.max_distance_edge_point[1], color='orange', marker='.', label='E'
                , markersize=10, clip_on=False)
        ax.plot(self.max_distance_boundary_point[0], self.max_distance_boundary_point[1], '.'
                , label=self.__get_point_label('MB', self.max_distance_boundary_point), markersize=10, clip_on=False)
        ax.plot(*self.__get_line(boundary_point=self.max_distance_boundary_point, p=self.max_distance_edge_point), 'k--'
                , label=f'Max Distance = {round(self.max_distance, self.decimal_places_round)}', clip_on=False)

        ax.set_title(f'Distance: {self.distance}, of max: {round(100*self.norm_payload["max_dist_edge_dbf"], self.decimal_places_round)}%')
        ax.legend()

    def __get_line(self, boundary_point, p, num_data_points=50):
        # return coordinates describing a line between p and boundary_point
        m = (boundary_point[1] - p[1]) / (boundary_point[0] - p[0])
        c = boundary_point[1] - m * boundary_point[0]

        if p[0] < boundary_point[0]:
            x = np.linspace(p[0], boundary_point[0], num=num_data_points)
            return (x, m * x + c)
        else:
            x = np.linspace(boundary_point[0], p[0], num=num_data_points)
            return (x, m * x + c)

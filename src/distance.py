from scipy.optimize import minimize_scalar
import numpy as np
from abc import ABC
from typing import Callable

from src.utils.datatypes import Point, Bounds, DistanceOutput, NormOutput


class Distance(ABC):
    """ Determine the closest point on a decision boundary to a point P """

    decimal_places_round = 5 # class attribute so we can import in test assertions

    def __init__(self, p: Point, dbf: Callable, b: Bounds, norm: bool = False):
        # instance attributes
        self.p = p # point p
        self.dbf = dbf # decision boundary function f
        self.bounds = b # bounds for f
        self.norm = norm # bool denoting whether to normalise distances relative to max possible distance or not

    def __objective(self, x):
        """The objective is to minimise the returned function below; the Euclidean distance of a point X from P"""
        y = self._f(x)
        return self.euclidean_distance(x=x, y=y)

    def _f(self, x):
        try:
            f = self.dbf(x)
        except Exception as e:
            print(f'The decision boundary function could not be evaluated over the x values supplied owing to error: {e}')

        # cap the result by boundaries, requires conversion to array first
        f = np.array(f)
        f[f < self.bounds.y.lower] = self.bounds.y.lower
        f[f > self.bounds.y.upper] = self.bounds.y.upper
        return f

    def euclidean_distance(self, x, y):
        return np.sqrt((self.p.x - x) ** 2 + (self.p.y - y) ** 2)

    def execute(self) -> DistanceOutput:

        print(f'Finding nearest distance to point P: {self.p}')

        # find the closest point on the boundary to the point p
        closest_boundary_point_to_p = minimize_scalar(self.__objective, bounds=self.bounds.x_tuple, method='bounded')
        self.boundary_point = [closest_boundary_point_to_p.x, float(self._f(closest_boundary_point_to_p.x))]
        self.boundary_point = Point(round(self.boundary_point[0], 5), round(self.boundary_point[1], 5))

        print(f'Closest boundary point CB to P: {self.boundary_point}')

        self.distance = round(self.euclidean_distance(x=self.boundary_point.x, y=self.boundary_point.y), 5)

        print(f'Distance of P to CB: {self.distance}')

        if self.norm:
            self.norm_payload = self.get_norm_distances()
            print(f'Max distance from boundary point MB to edge point E: {self.norm_payload["max_dist_edge_dbf"]}')
            print(f'Closest distance as proportion of max distance: {100*self.norm_payload["dist_p_proportion"]}%')

        output: DistanceOutput = {
            'closest_point': self.boundary_point,
            'distance': round(self.distance, self.decimal_places_round),
            'norm': self.norm_payload if self.norm else None
        }
        return output

    def get_norm_distances(self, n:int = 50) -> NormOutput:

        edge_points = []
        closest_boundary_point_to_all_edge_points = []
        edge_distances = []

        edge_coords  = {'bottom': (list(np.linspace(self.bounds.x.lower, self.bounds.x.upper, n)), [self.bounds.y.lower] * n)
                         , 'top': (list(np.linspace(self.bounds.x.lower, self.bounds.x.upper, n)), [self.bounds.y.upper] * n)
                        , 'left': ([self.bounds.x.lower] * n, list(np.linspace(self.bounds.y.lower, self.bounds.y.upper, n)))
                       , 'right': ([self.bounds.x.upper] * n, list(np.linspace(self.bounds.y.lower, self.bounds.y.upper, n)))}

        # for each of the 4 edges our plot has...
        for edge, coords in edge_coords.items():
            # construct edge points co-ordinates saved as a list i.e [(x1,y1), (x2,y2),...]
            zipped_coords = list(zip(coords[0], coords[1]))
            for coord in zipped_coords:
                edge_points.append(coord)
                # redefine p so minimize_scalar looks at the right point
                self.p = Point(coord[0], coord[1])
                # for our new p, what is the closest boundary point
                closest_boundary_point_to_edge_point = minimize_scalar(self.__objective, bounds=tuple(self.bounds.x_tuple), method='bounded')
                # notice how we call the 'x' part of the object just made
                closest_boundary_point_to_all_edge_points.append([closest_boundary_point_to_edge_point.x, self._f(closest_boundary_point_to_edge_point.x)])
                edge_distances.append(self.__objective(closest_boundary_point_to_edge_point.x))

        self.max_distance_index = np.argmax(edge_distances)
        self.max_distance = edge_distances[self.max_distance_index]
        self.max_distance_edge_point = edge_points[self.max_distance_index]
        # what is the associated boundary point?
        self.max_distance_boundary_point = closest_boundary_point_to_all_edge_points[self.max_distance_index]

        norm_output: NormOutput = {
            'max_dist_edge_dbf': self.max_distance,
            'max_dist_edge_point': Point(x=self.max_distance_edge_point[0], y=self.max_distance_edge_point[1]),
            'max_dist_dbf_point': Point(x=self.max_distance_boundary_point[0], y=self.max_distance_boundary_point[1]),
            'dist_p_proportion': self.distance / self.max_distance
        }

        # round any floats
        for key, value in norm_output.items():
            if isinstance(value, float):
                norm_output[key] = round(value, self.decimal_places_round)

        return norm_output

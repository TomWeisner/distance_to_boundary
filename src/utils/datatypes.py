from typing import Dict, Tuple, Optional, List, TypedDict
from numbers import Number

class Point():
    """ stores a point in 2d """
    def __init__(self, x: Number, y: Number):
        self.x = float(x)
        self.y = float(y)
        assert isinstance(self.x, Number), f'dtype of supplied x={x} is {type(x).__name__}, but must be a number'
        assert isinstance(self.y, Number), f'dtype of supplied y={y} is {type(y).__name__}, but must be a number'

class BoundsAxis():
    """ stores the upper and lower limits of a single axis bounds, where upper must be >= lower """
    def __init__(self, t: Tuple[Number, Number]):
        self.lower = t[0]
        self.upper = t[1]

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, new_upper: Number) -> None:
        assert new_upper >= self.lower
        self._upper = new_upper
        self.extent = self.upper - self.lower


class Bounds():
    """ stores rectangular axis bounds in 2d """
    def __init__(self, bound_dict: Dict[str, Tuple[float, float]]):
        self.bound_dict = bound_dict

    @property
    def bound_dict(self):
        return self._bound_dict

    @bound_dict.setter
    def bound_dict(self, supplied_bound_dict: Dict[str, Tuple[float, float]]) -> None:
        self._bound_dict = supplied_bound_dict
        self.x = BoundsAxis(self.bound_dict['x'])
        self.y = BoundsAxis(self.bound_dict['y'])

        self.area: float = self.x.extent * self.y.extent

        # the minimize_scalar method requires a tuple for boundary conditions
        self.x_tuple = tuple([self.x.lower, self.x.upper])
        self.y_tuple = tuple([self.y.lower, self.y.upper])

class NormOutput(TypedDict):
    max_dist_edge_dbf: float # max distance from any edge point to the decision boundary function
    max_dist_edge_point: Point # edge point that is the max distance from the dbf
    max_dist_dbf_point: Point # dbf point that is the closest point to max_distance_edge_point
    dist_p_proportion: float # proportion (0 to 1) the point p's distance from the dbf is relative to max_dist_edge_dbf

class DistanceOutput(TypedDict):
    closest_point: Tuple[float, float] # 2d coordinates
    distance: float # euclidean distance of a point p to closest_point
    norm: Optional[NormOutput] # if norm=True when executing the Distance class
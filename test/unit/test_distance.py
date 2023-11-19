from src.distance import Distance
from src.utils.datatypes import Bounds, DistanceOutput, Point
from src.utils.dbfs import flat, diag, diag_sine, arch
from test.utils.utils import clock

dp = Distance.decimal_places_round

import numpy as np

bounds = Bounds(bound_dict={'x': [0, 1], 'y': [0, 1]})

@clock
def test_flat_norm_false():
    # flat line is a line going along y=0.8
    p = Point(x=0.1, y=0.1)
    output = Distance(p=p, dbf=flat, b=bounds, norm=False).execute()
    assert output['closest_point'].x == 0.1 # x coordinate,
    assert output['closest_point'].y == 0.8 # then y value of line
    assert output['distance'] == round(0.8-0.1, dp) == 0.7
    assert output['norm'] is None

@clock
def test_flat_norm_true():
    # flat line is a line going along y=0.8
    p = Point(x=0.1, y=0.1)
    output = Distance(p=p, dbf=flat, b=bounds, norm=True).execute()
    assert output['closest_point'].x == 0.1 # x coordinate,
    assert output['closest_point'].y == 0.8 # then y value of line
    assert output['distance'] == round(0.8-0.1, dp) == 0.7
    assert output['norm'] is not None
    assert output['norm']['dist_p_proportion'] == round(0.7 / 0.8, dp)

@clock
def test_flat_norm_true_wide_bounds():
    # flat line is a line going along y=0.8
    p = Point(x=0.1, y=0.1)
    wide_bounds = Bounds(bound_dict={'x': [0, 2], 'y': [0, 2]})
    output = Distance(p=p, dbf=flat, b=wide_bounds, norm=True).execute()
    assert output['closest_point'].x == 0.1  # x coordinate,
    assert output['closest_point'].y == 0.8  # then y value of line
    assert output['distance'] == 0.7
    # as the edge wide_bounds.y.upper is 1.2 above the line this is the greatest distance from the boundary to an edge
    assert output['norm'] is not None
    assert output['norm']['dist_p_proportion'] == round(0.7/(wide_bounds.y.upper-0.8), 5)

@clock
def test_diag_on_the_line():
    # y=x
    p = Point(x=0.1, y=0.1)
    output = Distance(p=p, dbf=diag, b=bounds, norm=True).execute()
    assert output['closest_point'].x == 0.1  # x coordinate,
    assert output['closest_point'].y == 0.1  # then y value of line
    assert output['distance'] == 0
    assert output['norm'] is not None
    assert output['norm']['dist_p_proportion'] == 0

@clock
def test_diag_off_the_line():
    # y=x
    p = Point(x=0.3, y=0.8)
    D = Distance(p=p, dbf=diag, b=bounds, norm=True)
    output = D.execute()
    midway_dist = (0.8-0.3)/2
    midway_coord = min(p.x, p.y) + midway_dist
    assert midway_coord == 0.55
    midway_point = (midway_coord, midway_coord)
    assert output['closest_point'].x == midway_coord
    assert output['closest_point'].y == midway_coord
    D.p = p # reset the variable overwritten in calculation
    assert output['distance'] == round(D.euclidean_distance(x=midway_coord, y=midway_coord), dp)
    assert output['norm'] is not None
    # the furthest distance from an edge point to the dbf is a bottom right/top left corner point to the line
    # this has distance sqrt(2) / 2, so proportion = distance * 2 / sqrt(2) = distance * sqrt(2)
    assert output['norm']['dist_p_proportion'] == round(output['distance'] * np.sqrt(2), dp)

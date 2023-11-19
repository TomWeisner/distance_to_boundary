from src.distance import Distance
from src.utils.datatypes import Point, BoundsAxis, Bounds, NormOutput, DistanceOutput
from src.utils.dbfs import diag
from test.utils.utils import clock

import pytest

@clock
def test_point():
    test_x = 3
    test_y = 5.1
    p = Point(x=test_x, y=test_y)
    assert vars(p) == {'x': test_x, 'y': test_y}

@clock
def test_point_invalid():
    test_x = 'a' # must be a number
    test_y = 5.1
    with pytest.raises(ValueError, match="could not convert string to float: 'a'"):
        p = Point(x=test_x, y=test_y)

@clock
def test_boundsaxis():
    ba = BoundsAxis(t = [0, 1])
    assert ba.upper == 1
    assert ba.lower == 0
    assert ba.extent == 1 - 0

@clock
def test_bounds():
    input_bound_dict = {'x': [0, 1], 'y': [5, 8.1]}
    b = Bounds(bound_dict=input_bound_dict)
    assert isinstance(b.x, BoundsAxis)
    assert isinstance(b.y, BoundsAxis)
    assert round(b.area, 1) == round(b.x.extent * b.y.extent, 1) == round((1-0) * (8.1-5), 1) == 3.1

    assert b.x_tuple == tuple(input_bound_dict['x']) # list has been converted to tuple
    assert b.y_tuple == tuple(input_bound_dict['y'])

@clock
def test_norm_output():
    p = Point(x=0.1, y=0.1)
    input_bound_dict = {'x': [0, 1], 'y': [5, 8.1]}
    b = Bounds(bound_dict=input_bound_dict)
    output = Distance(p=p, dbf=diag, b=b, norm=True).execute()
    assert output['norm'] is not NormOutput
    normOutput = output['norm']
    for key, value in normOutput.items():
        if 'point' in key: assert isinstance(value, Point)
        else: assert isinstance(value, float)

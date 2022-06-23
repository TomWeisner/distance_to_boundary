from scipy.optimize import minimize_scalar
import numpy as np
import numbers

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


class Distance():
    """Determine the closest point on a decision boundary to a point P"""

    def __init__(self, p, dbf, b, norm=False, plot=False):
        self.p = p # point p
        self.dbf = dbf # decision boundary function f
        self.b = dict(b) # bounds for f
        self.norm = norm # bool denoting whether to normalise distances relative to max possible distance or not
        self.plot = plot # bool denoting whether to plot or not

        assert 'x' in self.b and 'y' in self.b


    def objective(self, x):
        """The objective is to minimise the returned function below; the Euclidean distance of a point X from P"""
        y = self.f(x)
        return self.euclidean_distance(x, y)

    def f(self, x):
        # run the dbf and save the result
        try:
            f = self.dbf(x)
        except Exception as e:
            print(f'The decision boundary function could not be evaluated over the x values supplied owing to error: {e}')

        # cap the result by boundaries, requires conversion to array first
        f = np.array(f)
        f[f < self.b['y'][0]] = self.b['y'][0]
        f[f > self.b['y'][1]] = self.b['y'][1]
        return f

    def g(self, boundary_point, p, num_data_points=50):
        # return coordinates describing a line between p and boundary_point
        m = (boundary_point[1] - p[1]) / (boundary_point[0] - p[0])
        c = boundary_point[1] - m * boundary_point[0]

        if p[0] < boundary_point[0]:
            x = np.linspace(p[0], boundary_point[0], num=num_data_points)
            return (x, m * x + c)
        else:
            x = np.linspace(boundary_point[0], p[0], num=num_data_points)
            return (x, m * x + c)

    def euclidean_distance(cls, x, y):
        return np.sqrt((cls.p[0] - x) ** 2 + (cls.p[1] - y) ** 2)

    def execute(self):

        print(f'Finding nearest distance to point P: {self.p}')

        # find the closest point on the boundary to the point p
        closest_boundary_point_to_p = minimize_scalar(self.objective, bounds=tuple(self.b['x']), method='bounded')
        self.boundary_point = [closest_boundary_point_to_p.x, float(self.f(closest_boundary_point_to_p.x))]
        self.boundary_point = (round(self.boundary_point[0], 5), round(self.boundary_point[1], 5))

        print(f'Closest boundary point CB to P: {self.boundary_point}')

        self.distance = round(self.euclidean_distance(x=self.boundary_point[0], y=self.boundary_point[1]), 5)

        print(f'Distance of P to CB: {self.distance}')

        if self.plot:
            ax = self.construct_figure()
        else:
            ax = None

        if self.norm:
            self.get_norm_distances(ax=ax)
            print(f'Max distance from boundary point MB to edge point E: {round(self.max_distance, 5)}')
            print(f'Closest distance as proportion of max distance: {round(100*self.norm_distance, 5)}%')



    def construct_figure(self):
        fig, ax = plt.subplots()
        ax.set_xlim((self.b['x'][0], self.b['x'][1]))
        ax.set_ylim((self.b['y'][0], self.b['y'][1]))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        x = np.arange(self.b['x'][0], self.b['x'][1] + 0.01, 0.005)
        ax.plot(self.p[0], self.p[1], 'r.', markersize=10, label='P')
        ax.plot(self.boundary_point[0], self.boundary_point[1], 'g.', label='CB', markersize=10)
        ax.plot(x, self.f(x), '-', color='grey', label='Boundary')
        ax.plot(*self.g(boundary_point=self.boundary_point, p=self.p), 'b--', label=f'Distance = {round(self.distance, 3)}')
        ax.legend()
        return ax


    def get_norm_distances(self, n=50, ax=None):

        edge_points = []
        closest_boundary_point_to_all_edge_points = []
        edge_distances = []

        edge_coords  = {'bottom': (list(np.linspace(self.b['x'][0], self.b['x'][1], n)), [self.b['y'][0]] * n)
                         , 'top': (list(np.linspace(self.b['x'][0], self.b['x'][1], n)), [self.b['y'][1]] * n)
                        , 'left': ([self.b['x'][0]] * n, list(np.linspace(self.b['y'][0], self.b['y'][1], n)))
                       , 'right': ([self.b['x'][1]] * n, list(np.linspace(self.b['y'][0], self.b['y'][1], n)))}

        # for each of the 4 edges our plot has...
        for edge, coords in edge_coords.items():
            # construct edge points co-ordinates saved as a list i.e [(x1,y1), (x2,y2),...]
            zipped_coords = list(zip(coords[0], coords[1]))
            for coord in zipped_coords:
                edge_points.append(coord)
                # redefine p so minimize_scalar looks at the right point
                self.p = coord
                # for our new p, what is the closest boundary point
                closest_boundary_point_to_edge_point = minimize_scalar(self.objective, bounds=tuple(self.b['x']), method='bounded')
                # notice how we call the 'x' part of the object just made
                closest_boundary_point_to_all_edge_points.append([closest_boundary_point_to_edge_point.x, self.f(closest_boundary_point_to_edge_point.x)])
                edge_distances.append(self.objective(closest_boundary_point_to_edge_point.x))

        self.max_distance_index = np.argmax(edge_distances)
        self.max_distance = edge_distances[self.max_distance_index]
        self.max_distance_edge_point = edge_points[self.max_distance_index]
        # what is the associated boundary point?
        self.max_distance_boundary_point = closest_boundary_point_to_all_edge_points[self.max_distance_index]

        # norm distance of our original point p
        self.norm_distance = round(self.distance / self.max_distance, 3)

        if self.plot:
            ax.plot(self.max_distance_edge_point[0], self.max_distance_edge_point[1], color='orange', marker='.', label='E'
                    , markersize=10, clip_on=False)
            ax.plot(self.max_distance_boundary_point[0], self.max_distance_boundary_point[1], '.', label='MB', markersize=10)
            ax.plot(*self.g(boundary_point=self.max_distance_boundary_point, p=self.max_distance_edge_point), 'k--'
                    , label=f'Max Distance = {round(self.max_distance, 3)}')

            ax.set_title(f'Distance: {self.distance}, of max: {round(100*self.norm_distance, 3)}%')
            ax.legend()

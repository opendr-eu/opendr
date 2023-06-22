# Copyright 2020-2023 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from closed_shape_2D import ClosedShape2D
from line import Line


class Polygon(ClosedShape2D):
    """
    Polygon Class comprised of a list of ordered vertices.
    A Polygon is considered a closed region, so the first and last vertices are connected by an edge.
    No need to duplicate the first vertex in order to close it.
    """

    def __init__(self, vertices=None, compute_bounding_box=True, opacity=1.0):
        """
        Constructor.

        :param vertices: (list|ndarray) Ordered list of [x, y] vertices comprising a closed polygon.
                                        Last edge connects the first and last vertices automatically,
                                        so no need to duplicate either of them.
        :param compute_bounding_box: (bool) Determines whether to compute a bounding rectangular box for this polygon.
                                            Used only because bounding boxes are themselves polygons, and must not get
                                            their own bounding box, as then they would infinitely recurse.
        :param opacity: (float: [0.0, 1.0]) Opacity level of the polygon. 0.0 means that it is totally transparent,
                                            while 1.0 is totally opaque. Used for line_intersect() to randomly determine
                                            if a line intersects it or not when opacity is set to a value other than 1.
        """

        super(Polygon, self).__init__()

        self.vertices = None

        self.boundingBox = compute_bounding_box

        self.opacity = opacity

        self.min = None
        self.max = None

        if vertices is not None:
            self.set_vertices(vertices, compute_bounding_box)

    def set_vertices(self, vertices, compute_bounding_box=True):
        """
        Sets the vertices of a polygon after being created and recomputes its bounding box if specified.

        :param vertices: (list|ndarray) Ordered list of vertices comprising a closed polygon. Last edge connects the
                                        first and last vertices automatically, so no need to duplicate either of them.
        :param compute_bounding_box: (bool) Determines whether to compute a bounding rectangular box for this polygon.
                                            Used only because bounding boxes are themselves polygons, and must not get
                                            their own bounding box, as then they would infinitely recurse.

        :return: None
        """

        if not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)

        self.vertices = vertices

        if compute_bounding_box:
            self._set_bounding_box()

    def _set_bounding_box(self):
        """
        Sets the polygon's bounding box from its minimum and maximum x and y values.

        :return: (Polygon) A polygon representing a bounding box rectangle completely enclosing its parent Polygon.
        """

        x_s = self.vertices[:, 0]
        y_s = self.vertices[:, 1]
        self.min_x = np.min(x_s)
        self.min_y = np.min(y_s)
        self.max_x = np.max(x_s)
        self.max_y = np.max(y_s)

        return self.get_bounding_box()

    def get_bounding_box(self):
        """
        Gets the polygon's bounding box.

        :return: (Polygon) A polygon representing a bounding box rectangle completely enclosing its parent Polygon.
        """

        return Polygon([[self.min_x, self.min_y],
                        [self.min_x, self.max_y],
                        [self.max_x, self.max_y],
                        [self.max_x, self.max_y]], compute_bounding_box=False)

    def get_perimeter(self):
        """
        Returns the perimeter of the polygon.

        :return: (float) Perimeter of the polygon.
        """

        # TODO: Implement
        raise NotImplementedError

    def get_area(self):
        """
        Returns the area of the polygon.

        :return: (float) Area of the polygon.
        """

        # TODO: Implement
        raise NotImplementedError

    def is_point_inside(self, point):
        """
        Check if a given point lies inside the closed polygon.

        :return: (bool) True if the point lies inside the polygon, False otherwise
        """

        # TODO: Implement
        raise NotImplementedError

    def intersection_with_line(self, line):
        """
        Check if a given line segment intersects the polygon. Takes into account the polygon's opacity.

        :param line: (Line) A line segment to check whether it intersects the polygon.

        :return: (point|None) A point of the intersection closest to the line's first point p1 if the line segment
                              intersects the polygon's edges. None if it doesn't intersect.
        """

        if self.opacity < 0:
            return None

        if self.opacity < 1.0:
            reflection_prob = np.random.uniform(0.0, 1.0)
            if reflection_prob > self.opacity:
                return None

        # If polygon has more vertices than a rectangle
        if self.vertices.shape[0] > 4:
            # Check if line intersects bounding box, if not, don't even bother in checking in detail.
            if self.boundingBox:

                if self.get_bounding_box().intersection_with_line(line) is None:
                    return None

        min_p = None

        for i, v in enumerate(self.vertices):

            edge = Line(v, self.vertices[i - 1])

            p = line.intersects(edge)

            if p is not None:
                # Keep the point closest to the first point in the line
                if min_p is None or Line(p, line.p1).len < Line(min_p, line.p1).len:
                    min_p = p

        return min_p

    def __str__(self):
        """
        String representation of the polygon as a list of vertices for easier debugging and printing.

        :return: (string) String representation of the set of vertices.
        """

        vertex_str = "[Poly: {"

        for vertex in self.vertices:
            vertex_str += str(vertex) + ", "

        vertex_str = vertex_str[0:-2] + "}"

        vertex_str += ', Op.: '
        vertex_str += str(self.opacity)
        vertex_str += ']'

        return vertex_str

    def __repr__(self):
        """
        String representation of the polygon as a list of vertices for easier debugging and printing.

        :return: (string) String representation of the set of vertices.
        """

        return self.__str__()


if __name__ == "__main__":
    """
    Testing and Sample code
    """

    square = Polygon(np.array([[1., 1.], [1., 2.], [2., 2.], [2., 1.]]))

    ray = Line(np.array([0., 0.]), np.array([3., 2.]))
    intersection = square.intersection_with_line(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)

    ray = Line(np.array([0., 0.]), np.array([1., 0]))
    intersection = square.intersection_with_line(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)

    ray = Line(np.array([0., 1.]), np.array([10., 1]))
    intersection = square.intersection_with_line(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)

    ray = Line(np.array([0., 0.]), np.array([10., 10.]))
    intersection = square.intersection_with_line(ray)
    print("Square:", square, "intersects ray:", ray, "at", intersection)

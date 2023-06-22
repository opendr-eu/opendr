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

from map_simulator.geometry.primitives import Polygon
from map_simulator.map_obstacles.obstacle import Obstacle


class PolygonalObstacle(Obstacle):
    """
    Class for obstacles with polygonal shapes.
    """

    def __init__(self, config):
        """
        Create a PolygonalObstacle object.

        :param config: (dict) Configuration dictionary, with parameters:
                                  * "vertices": (list|np.ndarray) List of Ordered list of [x, y] vertices comprising a
                                                                  closed polygon. Last edge connects the first and last
                                                                  vertices automatically, so no need to duplicate either
                                                                  of them.
                                  * "opacity": (float) (Optional)[Default: 1.0] Opacity level of the polygon. 0.0 means
                                                       that it is totally transparent, while 1.0 is totally opaque.
                                                       Used for line_intersect() to randomly determine if a line
                                                       intersects it or not when opacity is set to a value other than 1.
        """

        super(PolygonalObstacle, self).__init__(config)

        vertices = config['vertices']

        if 'opacity' in config:
            opacity = float(config['opacity'])
            opacity = min(1.0, opacity)
            opacity = max(0.0, opacity)
        else:
            opacity = 1.0

        self._poly = Polygon(vertices=vertices, opacity=opacity)

    def get_bounding_box(self):
        """
        Returns the polygonal obstacle's bounding box, i.e.: the minimum and maximum x and y coordinates
        it occupies.

        :return: (tuple) Tuple of minimum and maximum x and y coordinates of the polygon (min_x, min_y, max_x, max_y).
        """
        return self._poly.min_x, self._poly.min_y, self._poly.max_x, self._poly.max_y

    def intersection_with_line(self, line):
        """
        Check if a given line segment intersects the polygon. Takes into account the polygon's opacity.

        :param line: (Line) A line segment to check whether it intersects the polygon.

        :return: (point|None) A point of the intersection closest to the line's first point p1 if the line segment
                              intersects the polygon's edges. None if it doesn't intersect.
        """

        return self._poly.intersection_with_line(line)

    def draw(self, axes):
        """
        Draws the polygon obstacle to the given Matplotlib axes.

        :param axes: (matplotlib.Axes) Axes in which the polygon obstacle will be drawn.

        :return: None
        """

        vertices = self._poly.vertices.transpose()
        axes.fill(vertices[0], vertices[1], edgecolor='tab:blue', hatch='////',
                  fill=False, alpha=self._poly.opacity)

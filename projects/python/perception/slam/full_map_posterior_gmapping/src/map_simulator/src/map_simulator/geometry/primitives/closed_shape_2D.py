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

from abc import ABCMeta, abstractmethod


class ClosedShape2D:
    """
    Abstract class to define a 2D Closed Shape.
    Defines the required interfaces for 2D shapes.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def get_bounding_box(self):
        """
        Method signature for returning the shape's bounding box, i.e.: a rectangle fully enclosing the shape.
        Must be implemented in child class!

        :return: (Polygon) A polygon representing a bounding box rectangle completely enclosing its parent Polygon.
        """

        pass

    @abstractmethod
    def get_perimeter(self):
        """
        Method signature for returning the shape's perimeter.
        Must be implemented in child class!

        :return: (float) The shape's perimeter.
        """

        pass

    @abstractmethod
    def get_area(self):
        """
        Method signature for returning the shape's area.
        Must be implemented in child class!

        :return: (float) The shape's area.
        """

        pass

    @abstractmethod
    def is_point_inside(self, point):
        """
        Method signature for checking if a given point lies inside the shape.

        :return: (bool) True if the point lies inside the shape, False otherwise
        """

        pass

    @abstractmethod
    def intersection_with_line(self, line):
        """
        Method signature for checking if a given line segment intersects the shape.
        Must be implemented in child class!

        :param line: (Line) A line segment to check whether it intersects the shape.

        :return: (point|None) A point of the intersection closest to the line's first point p1 if the line segment
                              intersects the polygon's edges. None if it doesn't intersect.
        """

        pass

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


class Obstacle:
    """
    Abstract class for creating 2D map obstacles.
    Defines the required interfaces for obstacles.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, config):
        """
        Used for initializing the abstract class and also performing argument verifications for child classes.

        :param config: (dict) Dictionary containing configuration of the obstacle.
        """

        if not isinstance(config, dict):
            raise TypeError("Config must be a dictionary.")

    @abstractmethod
    def get_bounding_box(self):
        """
        Method signature for returning the obstacle's bounding box, i.e.: the minimum and maximum x and y coordinates
        it occupies.
        Must be implemented in child class!

        :return: (tuple) Tuple of minimum and maximum x and y coordinates of the obstacle (min_x, min_y, max_x, max_y).
        """

        pass

    @abstractmethod
    def intersection_with_line(self, line):
        """
        Method signature for returning the intersection point of the obstacle with a given line.
        Must be implemented in child class!

        :param line: (map_simulator.geometry.Line) Line with which the intersection is to be computed.

        :return: (np.ndarray) Intersection point between obstacle and line, closest to Line.p1 if they intersect.
                              None otherwise.
        """

        pass

    @abstractmethod
    def draw(self, axes):
        """
        Method signature for drawing the obstacle to the given Matplotlib axes.
        Must be implemented in child class!

        :param axes: (matplotlib.Axes) Axes in which the obstacle will be drawn.

        :return: None
        """

        pass

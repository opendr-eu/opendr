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

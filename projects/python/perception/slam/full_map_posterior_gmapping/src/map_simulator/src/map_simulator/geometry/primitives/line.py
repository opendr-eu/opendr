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


class Line:
    """
    Line segment class comprised of two endpoints
    """

    def __init__(self, p1=None, p2=None):
        """
        Constructor

        :param p1: First point of the Line segment.
        :param p2: Second point of the Line segment.
        """

        self.p1 = p1
        self.p2 = p2

        self.len = self._length()
        self.slope = self._slope()

    def _slope(self):
        """
        Returns the slope of the line segment

        :return: (float) Slope of the line segment
        """

        diff = self.p2 - self.p1
        return np.arctan2(diff[1], diff[0])

    def _length(self):
        """
        Returns the length of the line segment

        :return: (float) Length of the line segment
        """

        diff = self.p2 - self.p1
        return np.sqrt(np.dot(diff, diff))

    def intersects(self, line2, outside_segments=False):
        """
        Detects whether this line intersects another line segment.
        Algorithm from:
            https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect#answer-565282

        :param line2: (Line) Second line to determine the intersection with
        :param outside_segments: (bool) Determines whether the intersection needs to fall within the two line segments
                                 in order to count. If True, then the lines are considered of infinite length.

        :return: (ndarray) The point where the two lines intersect each other. If collinear, then the point closest
                            to self.p1. Returns None if the lines don't intersect.
        """

        p = self.p1
        q = line2.p1
        r = self.p2 - p
        s = line2.p2 - q

        denominator = np.cross(r, s)
        q_p = q - p
        q_pxr = np.cross(q_p, r)
        r_len = np.dot(r, r)

        # Parallel
        if denominator == 0:
            # Collinear
            if q_pxr == 0:
                t0 = np.dot(q_p, r) / r_len
                t1 = t0 + np.dot(s, r) / r_len

                sr = np.dot(s, r)
                if sr < 0:
                    tmp = t0
                    t0 = t1
                    t1 = tmp

                if (0 <= t1 and 1 >= t0) or outside_segments:
                    return p + t0 * r

            # else:
            # Parallel and not intersecting

        # Not Parallel
        else:
            t = np.cross(q_p, s) / denominator
            u = q_pxr / denominator

            if (0 <= t <= 1 and 0 <= u <= 1) or outside_segments:
                return p + t * r

        # Lines do not intersect
        return None

    def is_parallel(self, line2):
        """
        Determines whether this line is parallel to another line

        :param line2: (Line) Second line to check parallelism with

        :return: (bool) True if lines are parallel, False otherwise
        """
        return self.slope == line2.slope

    def __str__(self):
        """
        String representation

        :return: A serialized string for more convenient printing and debugging
        """

        return "[Line: {" + str(self.p1) + ", " + str(self.p2) + "}]"

    def __repr__(self):
        """
        String representation

        :return: A serialized string for more convenient printing and debugging
        """

        return self.__str__()

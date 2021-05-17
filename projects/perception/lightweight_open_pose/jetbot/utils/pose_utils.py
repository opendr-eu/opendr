# Copyright 2020-2021 OpenDR European Project
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


def calculate_horizontal_offset(pose, width):
    """
    Calculates the relative distance of a pose from the center of the image
    @param pose: the pose for which we want to calculate the offset
    @param width: the width of input image for normalizing the offset
    @return: a number between -0.5 and 0.5 that represents the horizontal offset of the pose
    """
    pose_center = np.mean(pose[pose != -1].reshape((-1, 2)), 0)
    offset_x = (pose_center[0] / width) - 0.5
    return offset_x


def calculate_body_area(pose, image_height, image_width):
    """
    Returns the (approximate) area that a human pose covers on a frame.
    @param pose: the human pose
    @param image_height: height of the input image
    @param image_width: width of the input image
    @return: area covered by the human pose (number between 0 and 1)
    """
    coords = pose.data
    coords = coords[coords != -1]
    if len(coords) % 2 == 0:
        coords = coords.reshape((-1, 2))
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        height = (max_y - min_y) / image_height
        width = (max_x - min_x) / image_width
        return height * width
    else:
        return 0.5


def calculate_upper_body_height(pose, image_height):
    """
    Calculates the relative position of the upper body in the frame
    @param pose: pose for which we want to calculate the upper body position
    @param image_height: image height for normalizing the upper body position
    @return: A number between 0 ... 1 to indicate the upper body position
    """
    # Gather keypoints from upper body
    kpts = ['r_sho', 'l_sho', 'nose', 'l_eye', 'r_eye', 'neck']
    heights = [pose[x][1] for x in kpts if pose[x][1] != -1]
    # Calculate mean and normalize according to image height
    height = np.mean(heights) / image_height
    return height

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


def initialize_webots_setup(pose_robot, setup='standing_human_close'):

    if 'standing' in setup:
        pedestrian = pose_robot.robot.getFromDef('HUMAN_SITTING')
        scale_field = pedestrian.getField('scale')
        scale_field.setSFVec3f([1e-6, 1e-6, 1e-6])
        pedestrian = pose_robot.robot.getFromDef('HUMAN_STANDING')
        scale_field = pedestrian.getField('scale')
        scale_field.setSFVec3f([0.12, 0.12, 0.12])
    else:
        pedestrian = pose_robot.robot.getFromDef('HUMAN_SITTING')
        scale_field = pedestrian.getField('scale')
        scale_field.setSFVec3f([0.12, 0.12, 0.12])
        pedestrian = pose_robot.robot.getFromDef('HUMAN_STANDING')
        scale_field = pedestrian.getField('scale')
        scale_field.setSFVec3f([1e-6, 1e-6, 1e-6])

    robot = pose_robot.robot.getFromDef('ROBOT')
    translation_field = robot.getField('translation')
    rotation_field = robot.getField('rotation')
    if 'near_center' in setup:
        translation_field.setSFVec3f([0, 0.095, 3])
    elif 'far_center' in setup:
        translation_field.setSFVec3f([0, 0.095, 6])
    elif 'near_offset' in setup:
        translation_field.setSFVec3f([-1, 0.095, 3])
    elif 'far_other_offset' in setup:
        translation_field.setSFVec3f([3, 0.095, 7])
    elif 'far_offset' in setup:
        translation_field.setSFVec3f([-3, 0.095, 7])

    if 'front' in setup:
        rotation_field.setSFRotation([0.5773519358512958, -0.5773519358512958, -0.5773469358518515, -2.094395307179586])
    elif 'rear' in setup:
        rotation_field.setSFRotation([0.9351140830166266, -0.2505630222442344, -0.25055902224387927, -1.6378353071795866])

    pose_robot.step()

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

import argparse
import cv2
import time
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.perception.pose_estimation import draw, get_bbox

from numpy import std


def wave_detector(frame_list_):
    """
    The wave detector works by first detecting the left or right wrist keypoints,
    then checking if the keypoints are higher than the neck keypoint, which empirically produces
    a more natural wave gesture, and finally calculating the standard deviation of these keypoints
    on the x-axis over the last frames. If this deviation is higher than a threshold it assumes
    that the hand is making a waving gesture.

    :param frame_list_: A list where each element is the list of poses for a frame
    :type frame_list_: list
    :return: A dict where each key is a pose ID and the value is whether a wave gesture is
    detected, 1 for waving, 0 for not waving, -1 can't detect gesture
    :rtype: dict
    """
    pose_waves_ = {}  # pose_id: waving (waving = 1, not waving = 0, can't detect = -1)
    # Loop through pose ids in last frame to check for waving in each one
    for pose_id_ in frame_list_[-1].keys():
        pose_waves_[pose_id_] = 0
        # Get average position of wrists, get list of wrists positions on x-axis
        r_wri_avg_pos = [0, 0]
        l_wri_avg_pos = [0, 0]
        r_wri_x_positions = []
        l_wri_x_positions = []
        for frame in frame_list_:
            try:
                if frame[pose_id_]["r_wri"][0] != -1:
                    r_wri_avg_pos += frame[pose_id_]["r_wri"]
                    r_wri_x_positions.append(frame[pose_id_]["r_wri"][0])
                if frame[pose_id_]["l_wri"][0] != -1:
                    l_wri_avg_pos += frame[pose_id_]["l_wri"]
                    l_wri_x_positions.append(frame[pose_id_]["l_wri"][0])
            except KeyError:  # Couldn't find this pose_id_ in previous frames
                pose_waves_[pose_id_] = -1
                continue
        r_wri_avg_pos = [r_wri_avg_pos[0] / len(frame_list_), r_wri_avg_pos[1] / len(frame_list_)]
        l_wri_avg_pos = [l_wri_avg_pos[0] / len(frame_list_), l_wri_avg_pos[1] / len(frame_list_)]
        r_wri_x_positions = [r_wri_x_positions[i] - r_wri_avg_pos[0] for i in range(len(r_wri_x_positions))]
        l_wri_x_positions = [l_wri_x_positions[i] - l_wri_avg_pos[0] for i in range(len(l_wri_x_positions))]

        pose_ = None  # NOQA
        if len(frame_list_) > 0:
            pose_ = frame_list_[-1][pose_id_]
        else:
            pose_waves_[pose_id_] = -1
            continue
        r_wri_height, l_wri_height = r_wri_avg_pos[1], l_wri_avg_pos[1]
        nose_height, neck_height = pose_["nose"][1], pose_["neck"][1]

        if nose_height == -1 or neck_height == -1:
            # Can't detect upper pose_ (neck-nose), can't assume waving
            pose_waves_[pose_id_] = -1
            continue
        if r_wri_height == 0 and l_wri_height == 0:
            # Can't detect wrists, can't assume waving
            pose_waves_[pose_id_] = -1
            continue

        # Calculate the standard deviation threshold based on the distance between neck and nose to get proportions
        # The farther away the pose is the smaller the threshold, as the standard deviation would be smaller due to
        # the smaller pose
        distance = neck_height - nose_height
        std_threshold = 5 + ((distance - 50) / (200 - 50))*10

        # Check for wrist movement over multiple frames
        # Wrist movement is determined from wrist x position standard deviation
        r_wrist_movement_detected = False
        l_wrist_movement_detected = False
        r_wri_x_pos_std, l_wri_x_pos_std = 0, 0  # NOQA
        if r_wri_height < neck_height:
            if len(r_wri_x_positions) > len(frame_list_) / 2:
                r_wri_x_pos_std = std(r_wri_x_positions)
                if r_wri_x_pos_std > std_threshold:
                    r_wrist_movement_detected = True
        if l_wri_height < neck_height:
            if len(l_wri_x_positions) > len(frame_list_) / 2:
                l_wri_x_pos_std = std(l_wri_x_positions)
                if l_wri_x_pos_std > std_threshold:
                    l_wrist_movement_detected = True

        if r_wrist_movement_detected:
            pose_waves_[pose_id_] = 1
        elif l_wrist_movement_detected:
            pose_waves_[pose_id_] = 1
    return pose_waves_


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img_ = self.cap.read()
        if not was_read:
            raise StopIteration
        return img_


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    args = parser.parse_args()

    onnx, device, accelerate = args.onnx, args.device, args.accelerate
    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride, half_precision=half_precision)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    if onnx:
        pose_estimator.optimize()

    # Use the first camera available on the system
    image_provider = VideoReader(0)
    fps = 0
    try:
        counter = 0
        frame_list = []
        for img in image_provider:

            start_time = time.perf_counter()

            # Perform inference
            poses = pose_estimator.infer(img)
            # convert to dict with pose id as key for convenience
            poses = {k: v for k, v in zip([poses[i].id for i in range(len(poses))], poses)}

            pose_waves = {}
            if len(poses) > 0:
                frame_list.append(poses)
                # Keep poses of last fps/2 frames (half a second of poses)
                if fps != 0:
                    if len(frame_list) > int(fps/2):
                        frame_list = frame_list[1:]
                else:
                    if len(frame_list) > 15:
                        frame_list = frame_list[1:]
                pose_waves = wave_detector(frame_list)

            end_time = time.perf_counter()
            fps = 1.0 / (end_time - start_time)

            for pose_id, pose in poses.items():
                draw(img, pose)
                x, y, w, h = get_bbox(pose)
                if pose_waves[pose_id] == 1:
                    x, y, w, h = get_bbox(pose)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, "Waving", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2, cv2.LINE_AA)

                if pose_waves[pose_id] == 0:
                    x, y, w, h = get_bbox(pose)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Not waving", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 0, 255), 2, cv2.LINE_AA)
                if pose_waves[pose_id] == -1:
                    x, y, w, h = get_bbox(pose)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                    cv2.putText(img, "Can't detect waving", (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # Wait a few frames for FPS to stabilize
            if counter > 5:
                cv2.putText(img, "FPS: %.2f" % (fps,), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Result', img)
            cv2.waitKey(1)
            counter += 1
    except KeyboardInterrupt as e:
        print(e)
        print("Average inference fps: ", fps)

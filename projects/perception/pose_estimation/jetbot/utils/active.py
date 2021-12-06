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

# This file provides a hotpatch to support an early version of active perception as a demonstrator, it will be removed
# in a future version of OpenDR toolkit


import torch
import cv2
from opendr.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import Image, extract_keypoints,\
    normalize, pad_width, group_keypoints, track_poses, FilteredPose, Pose, LightweightOpenPoseLearner
import numpy as np


def infer_active(self, img, upsample_ratio=4, track=True, smooth=True):
    """
    This method is used to perform pose estimation on an image.

    :param img: image to run inference on
    :rtype img: engine.data.Image class object
    :param upsample_ratio: Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing,
        defaults to 4
    :type upsample_ratio: int, optional
    :param track: If True, infer propagates poses ids from previous frame results to track poses, defaults to 'True'
    :type track: bool, optional
    :param smooth: If True, smoothing is performed on pose keypoints between frames, defaults to 'True'
    :type smooth: bool, optional
    :return: Returns a list of engine.target.Pose objects, where each holds a pose, or returns an empty list if no
        detections were made.
    :rtype: list of engine.target.Pose objects
    """
    if not isinstance(img, Image):
        img = Image(img)
    img = img.numpy()

    height, width, _ = img.shape
    scale = self.base_height / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, self.img_mean, self.img_scale)
    min_dims = [self.base_height, max(scaled_img.shape[1], self.base_height)]
    padded_img, pad = pad_width(scaled_img, self.stride, self.pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if self.device == "cuda":
        tensor_img = tensor_img.cuda()
        if self.half:
            tensor_img = tensor_img.half()

    if self.ort_session is not None:
        stages_output = self.ort_session.run(None, {'data': np.array(tensor_img.cpu())})
        stage2_heatmaps = torch.tensor(stages_output[-2])
        stage2_pafs = torch.tensor(stages_output[-1])
    else:
        if self.model is None:
            raise UserWarning("No model is loaded, cannot run inference. Load a model first using load().")
        if self.model_train_state:
            self.model.eval()
            self.model_train_state = False
        stages_output = self.model(tensor_img)
        stage2_heatmaps = stages_output[-2]
        stage2_pafs = stages_output[-1]

    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    if self.half:
        heatmaps = np.float32(heatmaps)
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    if self.half:
        pafs = np.float32(pafs)
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    num_keypoints = 18
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                 total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        if smooth:
            pose = FilteredPose(pose_keypoints, pose_entries[n][18])
        else:
            pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    if track:
        track_poses(self.previous_poses, current_poses, smooth=smooth)
        self.previous_poses = current_poses

    heatmaps = np.exp(heatmaps * 4)
    heatmaps = heatmaps / np.sum(heatmaps, -1, keepdims=True)

    return heatmaps, current_poses


# Hotpatch pose detector to support active perception
# This will not be needed in a future version of OpenDR toolkit
setattr(LightweightOpenPoseLearner, 'infer_active', infer_active)

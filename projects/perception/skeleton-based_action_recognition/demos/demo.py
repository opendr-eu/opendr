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
import os
import torch
import cv2
import time
import pandas as pd
from typing import Dict
from tqdm import tqdm


# opendr imports
from opendr.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from opendr.engine.data import Image
from opendr.engine.target import Pose
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import draw
import argparse
from opendr.perception.pose_estimation.lightweight_open_pose.filtered_pose import FilteredPose
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import track_poses
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.keypoints import \
    extract_keypoints, group_keypoints
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.val import \
    convert_to_coco_format, run_coco_eval, normalize, pad_width
from opendr.perception.skeleton_based_action_recognition.progressive_spatio_temporal_gcn_learner import \
    ProgressiveSpatioTemporalGCNLearner
from opendr.perception.skeleton_based_action_recognition.spatio_temporal_gcn_learner import SpatioTemporalGCNLearner


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::>>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966 >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0 >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)


def pre_normalization(data, zaxis=[0, 1], xaxis=[2, 5]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('pad the null frames with the previous frames')
    for i_s, skeleton in enumerate(tqdm(s)):  # pad
        if skeleton.sum() == 0:
            print(i_s, ' has no skeleton')
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            if person[0].sum() == 0:
                index = (person.sum(-1).sum(-1) != 0)
                tmp = person[index].copy()
                person *= 0
                person[:len(tmp)] = tmp
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    if person[i_f:].sum() == 0:
                        rest = len(person) - i_f
                        num = int(np.ceil(rest / i_f))
                        pad = np.concatenate([person[0:i_f] for _ in range(num)], 0)[:rest]
                        s[i_s, i_p, i_f:] = pad
                        break

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        main_body_center = skeleton[0][:, 1:2, :].copy()
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            mask = (person.sum(-1) != 0).reshape(T, V, 1)
            s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print(
        'parallel the bone between right shoulder(jpt 2) and left shoulder(jpt 5) of the first person to the x axis')
    for i_s, skeleton in enumerate(tqdm(s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


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
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def infer3Dpose(pose_estimator, img, upsample_ratio=4, track=True, smooth=True):
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
    scale = pose_estimator.base_height / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, pose_estimator.img_mean, pose_estimator.img_scale)
    min_dims = [pose_estimator.base_height, max(scaled_img.shape[1], pose_estimator.base_height)]
    padded_img, pad = pad_width(scaled_img, pose_estimator.stride, pose_estimator.pad_value, min_dims)
    H = padded_img.shape[0]
    W = padded_img.shape[1]

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if pose_estimator.device == "cuda":
        tensor_img = tensor_img.cuda()
        if pose_estimator.half:
            tensor_img = tensor_img.half()

    if pose_estimator.ort_session is not None:
        stages_output = pose_estimator.ort_session.run(None, {'data': np.array(tensor_img.cpu())})
        stage2_heatmaps = torch.tensor(stages_output[-2])
        stage2_pafs = torch.tensor(stages_output[-1])
    else:
        if pose_estimator.model is None:
            raise UserWarning("No model is loaded, cannot run inference. Load a model first using load().")
        if pose_estimator.model_train_state:
            pose_estimator.model.eval()
            pose_estimator.model_train_state = False
        stages_output = pose_estimator.model(tensor_img)
        stage2_heatmaps = stages_output[-2]
        stage2_pafs = stages_output[-1]

    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    if pose_estimator.half:
        heatmaps = np.float32(heatmaps)
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    if pose_estimator.half:
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
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * pose_estimator.stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * pose_estimator.stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    current_kptscores = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1  # changed by me
        keypoints_scores = np.ones((num_keypoints, 1), dtype=np.float32) * -1  # changed by me
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                keypoints_scores[kpt_id, 0] = round(all_keypoints[int(pose_entries[n][kpt_id]), 2], 2)  #
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        if smooth:
            pose = FilteredPose(pose_keypoints, pose_entries[n][18])
        else:
            pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
        current_kptscores.append(keypoints_scores)
    if track:
        track_poses(pose_estimator.previous_poses, current_poses, smooth=smooth)
        pose_estimator.previous_poses = current_poses
    return current_poses, current_kptscores, H, W


def tile(a, dim, n_tile):
    a = torch.from_numpy(a)
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    tiled_a = torch.index_select(a, dim, order_index)
    return tiled_a.numpy()


def pose2numpy(num_frames, poses_list, kptscores_list):
    C = 3
    T = 300
    V = 18
    M = 2  # num_person_in
    data_numpy = np.zeros((1, C, num_frames, V, M))
    skeleton_seq = np.zeros((1, C, T, V, M))
    for t in range(num_frames):
        for m in range(len(poses_list[t])):
            data_numpy[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)
            data_numpy[0, 2, t, :, m] = kptscores_list[t][m][:, 0]

    # if we have less than 300 frames, repeat frames to reach 300
    diff = T - num_frames
    if diff == 0:
        skeleton_seq = data_numpy
    while diff > 0:
        num_tiles = int(diff / num_frames)
        if num_tiles > 0:
            data_numpy = tile(data_numpy, 2, num_tiles+1)
            num_frames = data_numpy.shape[2]
            diff = T - num_frames
        elif num_tiles == 0:
            skeleton_seq[:, :, :num_frames, :, :] = data_numpy
            for j in range(diff):
                skeleton_seq[:, :, num_frames+j, :, :] = data_numpy[:, :, -1, :, :]
            break
    return skeleton_seq


MSR_CLASSES = pd.read_csv("./msr_daily_labels.csv", verbose=True, index_col=0).to_dict()["name"] #LH
NTU60_ClASSES = pd.read_csv("./ntu60_labels.csv", verbose=True, index_col=0).to_dict()["name"] #LH


def preds2label(confidence):  #LH
    k = 3
    class_scores, class_inds = torch.topk(confidence, k=3)
    labels = {NTU60_ClASSES[int(class_inds[j])]: float(class_scores[j].item())for j in range(k)}
    return labels


def draw_preds(frame, preds: Dict):  #LH
    base_skip = 40
    delta_skip = 30
    for i, (cls, prob) in enumerate(preds.items()):
        cv2.putText(
            frame,
            f"{prob:04.3f} {cls}",
            (10, base_skip + i * delta_skip),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),  # B G R,
            2,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    parser.add_argument('--video', type=str, default='./videos/input.mp4',
                        help='path to video file or camera id')  #
    parser.add_argument('--pose', type=str, default='./lightweight_openpose/openpose_default',
                        help='path to pose estimator model')
    parser.add_argument('--method', type=str, default='stgcn',
                        help='action detection method')
    parser.add_argument('--action', type=str, default='./stgcn/',
                        help='path to action detector model')
    parser.add_argument('--action_checkpoint', type=str, default='ntu_cv_stgcn_joint_49-29600',
                        help='action detector model name')
    parser.add_argument('--normalization', default=False)
    parser.add_argument('--save_dir', type=str, default='./videos',
                        help='path to save generated video')
    parser.add_argument('--video_name', type=str, default='output_video',
                        help='name of the output video')

    args = parser.parse_args()
    onnx, device = args.onnx, args.device
    accelerate = args.accelerate
    onnx, device, accelerate = args.onnx, args.device, args.accelerate
    if accelerate:
        stride = True
        stages = 0
        half_precision = True
    else:
        stride = False
        stages = 2
        half_precision = False

    # pose estimator
    pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride, half_precision=half_precision)
    pose_estimator.load(args.pose)

    # Action classifier
    if args.method == 'pstgcn':
        action_classifier = ProgressiveSpatioTemporalGCNLearner(device="cuda", dataset_name='kinetics',
                                                                topology=[5, 4, 5, 2, 3, 4, 3, 4])
    else:
        action_classifier = SpatioTemporalGCNLearner(device="cuda", dataset_name='kinetics', method_name=args.method)

    model_saved_path = args.action
    model_name = args.action_checkpoint
    action_classifier.load(model_saved_path, model_name)

    # Optimization
    if onnx:
        pose_estimator.optimize()
        action_classifier.optimize()

    image_provider = VideoReader(0)
    #image_provider = VideoReader(args.video)

    video_name = args.video_name
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    video_path = os.path.join(args.save_dir, video_name)
    counter, avg_fps = 0, 0
    poses_list = []
    kptscores_list = []
    for img in image_provider:
        start_time = time.perf_counter()
        poses, kptscores, H, W = infer3Dpose(pose_estimator, img)
        for pose in poses:
            draw(img, pose)
        # select two poses with highest confidence score
        if len(poses) > 2:
            selected_poses = []
            posescores = []
            for i in range(len(poses)):
                posescores.append(poses[i].confidence)
            sorted_idx = sorted(range(len(posescores)), key=lambda k: posescores[k])
            for i in range(2):
                selected_poses.append(poses[sorted_idx[i]])
            poses = selected_poses

        if len(poses) > 0:
            counter += 1
            print(counter)
            poses_list.append(poses)
            kptscores_list.append(kptscores)

        if counter > 300:
            for cnt in range(counter-300):
                poses_list.pop(0)
                kptscores_list.pop(0)
            counter = 300
        if counter > 0:
            skeleton_seq = pose2numpy(counter, poses_list, kptscores_list)
            if args.normalization:
                skeleton_seq = pre_normalization(skeleton_seq)

        prediction = action_classifier.infer(skeleton_seq)
        category_labels = preds2label(prediction.confidence)
        print(category_labels)
        draw_preds(img, category_labels)
        # Calculate a running average on FPS
        end_time = time.perf_counter()
        fps = 1.0 / (end_time - start_time)
        avg_fps = 0.8 * fps + 0.2 * fps
        # Wait a few frames for FPS to stabilize
        if counter > 5:
            img = cv2.putText(img, "FPS: %.2f" % (avg_fps,), (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                              1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Result', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    print("Average inference fps: ", avg_fps)



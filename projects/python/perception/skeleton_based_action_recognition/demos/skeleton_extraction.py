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
import os
import torch
import cv2
import pickle
from tqdm import tqdm

# opendr imports
from opendr.perception.pose_estimation import LightweightOpenPoseLearner
from opendr.engine.data import Image
from opendr.engine.target import Pose
import argparse
from opendr.perception.pose_estimation.lightweight_open_pose.filtered_pose import FilteredPose
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import track_poses
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.keypoints import \
    extract_keypoints, group_keypoints
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.val import normalize, pad_width


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
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        keypoints_scores = np.ones((num_keypoints, 1), dtype=np.float32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                keypoints_scores[kpt_id, 0] = round(all_keypoints[int(pose_entries[n][kpt_id]), 2], 2)  #
        if smooth:
            pose = FilteredPose(pose_keypoints, pose_entries[n][18])
        else:
            pose = Pose(pose_keypoints, pose_entries[n][18])

        pose_estimator.previous_poses = current_poses
        current_poses.append(pose)
        current_kptscores.append(keypoints_scores)
    if track:
        track_poses(pose_estimator.previous_poses, current_poses, smooth=smooth)
    return current_poses, current_kptscores


def tile(a, dim, n_tile):
    a = torch.from_numpy(a)
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    tiled_a = torch.index_select(a, dim, order_index)
    return tiled_a.numpy()


def pose2numpy(num_frames, poses_list, kptscores_list, num_channels=3):
    C = num_channels
    T = 300
    V = 18
    M = 2  # num_person_in
    data_numpy = np.zeros((1, C, num_frames, V, M))
    skeleton_seq = np.zeros((1, C, T, V, M))
    for t in range(num_frames):
        for m in range(len(poses_list[t])):
            data_numpy[0, 0:2, t, :, m] = np.transpose(poses_list[t][m].data)
            if C == 3:
                data_numpy[0, 2, t, :, m] = kptscores_list[t][m][:, 0]

    # if we have less than 300 frames, repeat frames to reach 300
    diff = T - num_frames
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


def select_2_poses(poses):
    selected_poses = []
    energy = []
    for i in range(len(poses)):
        s = poses[i].data[:, 0].std() + poses[i].data[:, 1].std()
        energy.append(s)
    energy = np.array(energy)
    index = energy.argsort()[::-1][0:2]
    for i in range(len(index)):
        selected_poses.append(poses[index[i]])
    return selected_poses


def data_gen(args, pose_estimator, out_path, benchmark, part):

    training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    training_cameras = [2, 3]
    if args.ignored_sample_path is not None:
        with open(args.ignored_sample_path, 'r') as f:
            ignored_samples = [line.strip() + '.avi' for line in f.readlines()]
    else:
        ignored_samples = []

    sample_names = []
    sample_labels = []
    for filename in os.listdir(args.videos_path):
        if filename in ignored_samples:
            continue
        action_class = int(filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not istraining
        else:
            raise ValueError()

        if issample:
            sample_name = (filename.split('.'))[0].split('_')[0]
            sample_names.append(sample_name)
            sample_labels.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_names, list(sample_labels)), f)

    skeleton_data = np.zeros((len(sample_labels), args.num_channels, 300, 18, 2), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_names)):
        video_path = os.path.join(args.videos_path, s + '_rgb.avi')
        image_provider = VideoReader(video_path)
        counter = 0
        poses_list = []
        kptscores_list = []
        pose_estimator.previous_poses = []
        for img in image_provider:
            poses, kptscores = infer3Dpose(pose_estimator, img)
            if len(poses) > 2:
                # select two poses with highest energy
                poses = select_2_poses(poses)
            if len(poses) > 0:
                counter += 1
                poses_list.append(poses)
                kptscores_list.append(kptscores)
        if counter > 300:
            for cnt in range(counter - 300):
                poses_list.pop(0)
                kptscores_list.pop(0)
            counter = 300
        if counter > 0:
            skeleton_seq = pose2numpy(counter, poses_list, kptscores_list, args.num_channels)
            skeleton_data[i, :, :, :, :] = skeleton_seq

    np.save('{}/{}_data_joint.npy'.format(out_path, part), skeleton_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", help="Use ONNX", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda")
    parser.add_argument("--accelerate", help="Enables acceleration flags (e.g., stride)", default=False,
                        action="store_true")
    parser.add_argument('--videos_path', type=str, default='/mnt/archive/nh/NTU60_RGB_Videos/nturgb+d_rgb',
                        help='path to video files')
    parser.add_argument("--num_channels", help="number of channels for each keypoint", default=2)
    parser.add_argument('--ignored_sample_path', default='./samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='/mnt/archive/nh/NTU60_Skeletons_LightweightOpenPose/2dSkeletons')
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

    # pose estimator
    pose_estimator = LightweightOpenPoseLearner(device=device, num_refinement_stages=stages,
                                                mobilenet_use_stride=stride, half_precision=half_precision)
    pose_estimator.download(path=".", verbose=True)
    pose_estimator.load("openpose_default")

    # Optimization
    if onnx:
        pose_estimator.optimize()

    benchmark = ['xview', 'xsub']
    part = ['train', 'val']

    for b in benchmark:
        for p in part:
            out_path = os.path.join(args.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            data_gen(args, pose_estimator, out_path, benchmark=b, part=p)

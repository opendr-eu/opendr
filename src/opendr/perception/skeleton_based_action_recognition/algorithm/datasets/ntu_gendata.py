"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""

import argparse
import pickle
from tqdm import tqdm
import numpy as np
import os
import math
import pandas
from pathlib import Path


training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300

NTU60_CLASSES = pandas.read_csv(Path(__file__).parent / 'ntu60_labels.csv', verbose=True, index_col=0).to_dict()["name"]


def read_skeleton(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []
            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)
    return skeleton_sequence


def read_xyz(file, max_body=2, num_joint=25):
    seq_info = read_skeleton(file)
    data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # C, T, V, M
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[:, n, j, m] = [v['x'], v['y'], v['z']]
                else:
                    pass
    return data


def fill_empty_frames(data):
    for s, skeleton in enumerate(tqdm(data)):
        if skeleton.sum() != 0:
            for p, person in enumerate(skeleton):
                if person.sum() != 0:
                    nonzero_idx = (person.sum(-1).sum(-1) != 0)
                    nonzero_frames = person[nonzero_idx].copy()
                    person = np.zeros(person.shape)
                    person[:len(nonzero_frames)] = nonzero_frames
                    for f, frame in enumerate(person):
                        if frame.sum() == 0:
                            if person[f:].sum() == 0:
                                rest = len(person) - f
                                num = int(np.ceil(rest / f))
                                pad = np.concatenate([person[0:f] for _ in range(num)], 0)[:rest]
                                data[s, p, f:] = pad
                                break
    return data


def skeleton_preprocess(data):
    N, M, T, V, C = data.shape
    # centralization
    for s, skeleton in enumerate(tqdm(data)):
        if skeleton.sum() != 0:
            center_joint = skeleton[0][:, 1:2, :].copy()
            for p, person in enumerate(skeleton):
                if person.sum() != 0:
                    mask = (person.sum(-1) != 0).reshape(T, V, 1)
                    data[s, p] = (data[s, p] - center_joint) * mask

    # parallelize the skeletons x and z axis
    z_axis = [0, 0, 1]
    x_axis = [1, 0, 0]
    for s, skeleton in enumerate(tqdm(data)):
        if skeleton.sum() != 0:
            bone = skeleton[0, 0, 1] - skeleton[0, 0, 0]
            perpendicular_axis = np.asarray(np.cross(bone, z_axis))
            if np.abs(z_axis).sum() > 1e-5 and np.abs(bone).sum() > 1e-5:
                bone_unit = bone/np.linalg.norm(bone)
                z_axis_unit = z_axis/np.linalg.norm(z_axis)
                vec = np.dot(bone_unit, z_axis_unit)
                angle = np.arccos(np.clip(vec, -1.0, 1.0))
            else:
                angle = 0
            if np.abs(perpendicular_axis).sum() > 1e-5 and np.abs(angle).sum() > 1e-5:
                axis = perpendicular_axis / np.linalg.norm(perpendicular_axis)
                q0 = math.cos(angle / 2.0)
                q1, q2, q3 = -axis * math.sin(angle / 2.0)
                rotation_map = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3), 2 * (q1*q3 - q0*q2)],
                                        [2 * (q1*q2 - q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)],
                                        [2 * (q3*q1 + q0*q2), 2 * (q3*q2 - q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
            else:
                rotation_map = np.eye(3)
            for p, person in enumerate(skeleton):
                if person.sum() != 0:
                    for f, frame in enumerate(person):
                        if frame.sum() != 0:
                            for j, joint in enumerate(frame):
                                data[s, p, f, j] = np.dot(rotation_map, joint)

    for s, skeleton in enumerate(tqdm(data)):
        if skeleton.sum() != 0:
            bone = skeleton[0, 0, 8] - skeleton[0, 0, 4]  # shoulders
            perpendicular_axis = np.asarray(np.cross(bone, x_axis))
            if np.abs(x_axis).sum() > 1e-5 and np.abs(bone).sum() > 1e-5:
                bone_unit = bone/np.linalg.norm(bone)
                x_axis_unit = x_axis/np.linalg.norm(x_axis)
                vec = np.dot(bone_unit, x_axis_unit)
                angle = np.arccos(np.clip(vec, -1.0, 1.0))
            else:
                angle = 0
            if np.abs(perpendicular_axis).sum() > 1e-5 and np.abs(angle).sum() > 1e-5:
                axis = perpendicular_axis / np.linalg.norm(perpendicular_axis)
                q0 = math.cos(angle / 2.0)
                q1, q2, q3 = -axis * math.sin(angle / 2.0)
                rotation_map = np.array([[q0**2 + q1**2 - q2**2 - q3**2, 2 * (q1*q2 + q0*q3), 2 * (q1*q3 - q0*q2)],
                                        [2 * (q1*q2 - q0*q3), q0**2 - q1**2 + q2**2 - q3**2, 2 * (q2*q3 + q0*q1)],
                                        [2 * (q3*q1 + q0*q2), 2 * (q3*q2 - q0*q1), q0**2 - q1**2 - q2**2 + q3**2]])
            else:
                rotation_map = np.eye(3)
            for p, person in enumerate(skeleton):
                if person.sum() != 0:
                    for f, frame in enumerate(person):
                        if frame.sum() != 0:
                            for j, joint in enumerate(frame):
                                data[s, p, f, j] = np.dot(rotation_map, joint)

    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    if ignored_sample_path is not None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data

    data = np.transpose(fp, [0, 4, 2, 3, 1])  # N, M, T, V, C
    filled_data = fill_empty_frames(data)
    preprocessed_data = skeleton_preprocess(filled_data)
    fp = np.transpose(preprocessed_data, [0, 4, 2, 3, 1])  # N, C, T, V, M
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument('--data_path', default='./data/nturgbd_raw/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path',
                        default='./algorithm/datasets/ntu_samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='./data/ntu/')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)

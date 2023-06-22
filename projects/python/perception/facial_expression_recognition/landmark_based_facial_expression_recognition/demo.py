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


import os
import cv2
import argparse
import numpy as np
import torch
import pandas

from opendr.perception.facial_expression_recognition import ProgressiveSpatioTemporalBLNLearner
from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.\
    algorithm.datasets.landmark_extractor import landmark_extractor
from opendr.perception.facial_expression_recognition.landmark_based_facial_expression_recognition.\
    algorithm.datasets.gen_facial_muscles_data import gen_muscle_data


def preds2label(labels_csv_path, confidence):
    k = 3
    class_scores, class_inds = torch.topk(confidence, k=k)
    expression_classes = pandas.read_csv(labels_csv_path, verbose=True, index_col=0).to_dict()["name"]
    labels = {expression_classes[int(class_inds[j])]: float(class_scores[j].item())for j in range(k)}
    return labels


def getFrame(vidcap, sec, framespth, count):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(framespth, "frame" + str(count) + ".jpg"), image)
    return hasFrames


def tile(a, dim, n_tile):
    a = torch.from_numpy(a)
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    tiled = torch.index_select(a, dim, order_index)
    return tiled.numpy()


def data_normalization(data):
    data = torch.from_numpy(data)
    N, V, C, T, M = data.size()
    data = data.permute(0, 2, 3, 1, 4).contiguous().view(N, C, T, V, M)
    # remove the first 17 points
    data = data[:, :, :, 17:, :]
    N, C, T, V, M = data.size()
    # normalization
    for n in range(N):
        for t in range(T):
            for v in range(V):
                data[n, :, t, v, :] = data[n, :, t, v, :] - data[n, :, t, 16, :]
    return data.numpy()


def data_gen(landmark_path, num_frames, num_landmarks, num_dim, num_faces, model_name):
    if os.path.exists(landmark_path):
        root, _, files = os.walk(landmark_path)
        T = len(files)
        sample_numpy = np.zeros((1, num_landmarks, num_dim, num_frames, num_faces))
        if T > num_frames or model_name in ['pstbln_casia', 'pstbln_ck+']:  # num_frames = 5
            for j in range(num_frames-1):
                if os.path.isfile(landmark_path + str(T - j - 1) + '.npy'):
                    sample_numpy[0, :, :, -1 - j, 0] = np.load(landmark_path + str(T - j - 1) + '.npy')
            for j in range(T):
                if os.path.isfile(landmark_path + str(j) + '.npy'):
                    sample_numpy[0, :, :, 0, 0] = np.load(landmark_path + str(j) + '.npy')
                    break
        elif T < num_frames or model_name in ['pstbln_afew']:  # num_frames = 300
            sample_numpy = np.zeros((1, num_landmarks, num_dim, T, num_faces))
            for j in range(T):
                if os.path.isfile(landmark_path + str(j) + '.npy'):
                    sample_numpy[0, :, :, j, 0] = np.load(landmark_path + str(j) + '.npy')
            dif = num_frames - T
            num_tiles = int(dif / T)
            while dif > 0:
                if num_tiles == 0:
                    for k in range(dif):
                        sample_numpy[0, :, :, T + k, :] = sample_numpy[0, :, :, -1, :]
                elif num_tiles > 0:
                    sample_numpy = tile(sample_numpy[:, :, :, :T, 0], 3, num_tiles)
                T = sample_numpy.shape[3]
                dif = num_frames - T
                num_tiles = int(dif / T)
        else:
            for j in range(num_frames):
                if os.path.isfile(landmark_path + str(j) + '.npy'):
                    sample_numpy[:, :, :, j, 0] = np.load(landmark_path + str(j) + '.npy')
    return sample_numpy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video frame extractor')
    parser.add_argument("-i", "--video_folder", required=True, default='./input.mp4',
                        description="path to input video")
    parser.add_argument("-i", "--labels_csv_path", required=True, default='./labels.csv',
                        description="path to reference labels file")
    parser.add_argument("-p", "--shape_predictor", required=True,
                        default='./shape_predictor_68_face_landmarks.dat',
                        description="path to facial landmark predictor")
    parser.add_argument('--checkpoint_path', type=str, default='./pstbln',
                        help='path to trained classifier model')
    parser.add_argument('--model_name', type=str, default='pstbln_casia',
                        help='name of the pretrained model')
    parser.add_argument('--num_frames', type=int, default=5, help='the number of frames in each sequence')
    parser.add_argument("-i", "--output_data_path", required=True, default='./data',
                        description="path to save the generated muscles data")

    args = vars(parser.parse_args())

    # 1: extract video frames:
    video_path = args.video_folder
    frames_path = os.path.join(video_path, 'frames_folder')
    if not os.path.exists(frames_path):
        os.makedirs(frames_path)
    vidcap = cv2.VideoCapture(video_path)
    sec = 0
    frameRate = 0.5  # it captures frames every 0.5 second
    count = 0
    success = getFrame(vidcap, sec, frames_path, count)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(vidcap, sec, frames_path, count)

    # 2: extract landmarks from each frame:
    landmarks_path = os.path.join(frames_path, 'landmarks_folder')
    if not os.path.exists(landmarks_path):
        os.makedirs(landmarks_path)
    for root, _, files in os.walk(frames_path):
        for file in files:
            if '.jpg' in file:
                imgpth = os.path.join(root, file)
                outpth = landmarks_path
                frameidx = file.split(".")
                landmark_extractor(imgpth, landmarks_path + frameidx[0] + '.npy', args.shape_predictor)

    # 3: sequence numpy data generation from extracted landmarks and normalization:
    num_landmarks = 68
    num_dim = 2  # feature dimension for each facial landmark
    num_faces = 1  # number of faces in each frame
    num_frames = args.num_frames
    model_name = args.model_name
    muscle_path = args.output_data_path
    numpy_data = data_gen(landmarks_path, num_frames, num_landmarks, num_dim, num_faces, model_name)
    norm_data = data_normalization(numpy_data)
    muscle_data = gen_muscle_data(norm_data, muscle_path)

    if args.model_name == 'pstbln_ck+':
        num_point = 303
        num_class = 7
    elif args.model_name == 'pstbln_casia':
        num_point = 309
        num_class = 6
    elif args.model_name == 'pstbln_afew':
        num_point = 312
        num_class = 7

    # inference
    expression_classifier = ProgressiveSpatioTemporalBLNLearner(device="cpu", dataset_name='AFEW', num_class=num_class,
                                                                num_point=num_point, num_person=1, in_channels=2,
                                                                blocksize=5, topology=[15, 10, 15, 5, 5, 10])
    model_saved_path = args.checkpoint_path
    expression_classifier.load(model_saved_path, model_name)
    prediction = expression_classifier.infer(muscle_data)
    category_labels = preds2label(args.labels_csv_path, prediction.confidence)
    print(category_labels)

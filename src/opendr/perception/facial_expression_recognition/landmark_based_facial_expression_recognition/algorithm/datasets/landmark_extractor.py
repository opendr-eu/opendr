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


import os
import imutils
from imutils import face_utils
import numpy as np
import argparse
import dlib
import cv2


def landmark_extractor(input_path, output_path, predictor_path):
    # dlib's face detector
    detector = dlib.get_frontal_face_detector()
    # the facial landmark predictor
    predictor = dlib.shape_predictor(predictor_path)
    image = cv2.imread(input_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        print(output_path)
        np.save(output_path, shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    return shape


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial landmark extractor')
    parser.add_argument('--dataset_name', default='CASIA')
    parser.add_argument("-p", "--shape_predictor", required=True,
                        default='./data/shape_predictor_68_face_landmarks.dat',
                        description="path to facial landmark predictor")
    parser.add_argument("-i", "--frames_folder", required=True, default='./data/CASIA/',
                        description="path to input image")
    parser.add_argument("-o", "--landmark_folder", required=True, default='./data/CASIA_landmarks/',
                        description="path to output")

    arg = vars(parser.parse_args())
    if not os.path.exists(arg.landmark_folder):
        os.makedirs(arg.landmark_folder)
    part = ['Train', 'Val']
    if arg.dataset_name == 'CASIA':
        num_subjects = len(os.listdir(arg.frames_folder))
        classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Disgust']
        for s in range(num_subjects):
            for c in range(classes):
                image_path = arg.frames_folder + '/{}/{}'.format(s, c)
                for root, _, files in os.walk(image_path):
                    for file in files:
                        if '.jpg' in file:
                            imgpth = os.path.join(root, file)
                            outpth = arg.landmark_folder + '/{}/{}'.format(s, c)
                            if not os.path.exists(outpth):
                                os.makedirs(outpth)
                            frameidx = file.split(".")
                            landmark_extractor(imgpth, outpth + frameidx[0] + '.npy', arg.shape_predictor)
    elif arg.dataset_name == 'CK+':
        num_subjects = len(os.listdir(arg.frames_folder))
        for s in range(num_subjects):
            image_path = arg.frames_folder + '/{}'.format(s)
            for _, dirs, _ in os.walk(image_path):
                for root, _, files in os.walk(dirs):
                    for file in files:
                        if '.jpg' in file:
                            imgpth = os.path.join(root, file)
                            outpth = arg.landmark_folder + '/{}/{}'.format(s, dirs)
                            if not os.path.exists(outpth):
                                os.makedirs(outpth)
                            frameidx = file.split(".")
                            landmark_extractor(imgpth, outpth + frameidx[0] + '.npy', arg.shape_predictor)
    elif arg.dataset_name == 'AFEW':
        classes = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Disgust', 'Neutral']
        for p in part:
            for c in classes:
                image_path = arg.frames_folder + '/{}/{}'.format(p, c)
                for _, dirs, _ in os.walk(image_path):
                    for root, _, files in os.walk(dirs):
                        for file in files:
                            if '.jpg' in file:
                                imgpth = os.path.join(root, file)
                                outpth = arg.landmark_folder + '/{}/{}/{}'.format(p, c, dirs)
                                if not os.path.exists(outpth):
                                    os.makedirs(outpth)
                                frameidx = file.split(".")
                                landmark_extractor(imgpth, outpth + frameidx[0] + '.npy', arg.shape_predictor)

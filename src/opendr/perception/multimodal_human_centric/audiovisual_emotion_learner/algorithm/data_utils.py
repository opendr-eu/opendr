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
import librosa
import numpy as np
import cv2
from opendr.perception.object_detection_2d import RetinaFaceLearner


def preprocess_video(video_path, target_time=3.6, input_fps=30, save_frames=15, target_im_size=224, device='cpu'):
    """
    This function preprocesses input video file: crops/pads it to desired target_time (match with audio),
    performs face detection and uniformly selects N frames
    Parameters
    ----------
    video_path : str
        path to video file.
    target_time : float, optional
        Target time of processed video file in seconds. The default is 3.6.
    input_fps : int, optional
        Frames Per Second of input video file. The default is 30.
    save_frames : int, optional
        Length of target frame sequence. The default is 15.
    target_im_size : int, optional
        Target width and height of each frame. The default is 224.

    Returns
    -------
    numpy_video: numpy.array
                 N frames as numpy array

    """

    learner = RetinaFaceLearner(backbone='resnet', device=device)

    if not os.path.exists('./retinaface_resnet'):
        learner.download(".", mode="pretrained")
    learner.load("./retinaface_resnet")

    def select_distributed(m, n): return [i*n//m + n//(2*m) for i in range(m)]

    cap = cv2.VideoCapture(video_path)
    framen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_time*input_fps > framen:
        skip_begin = int((framen - (target_time*input_fps)) // 2)
        for i in range(skip_begin):
            _, im = cap.read()

    framen = int(target_time*input_fps)
    frames_to_select = select_distributed(save_frames, framen)
    numpy_video = []
    frame_ctr = 0
    while True:
        ret, im = cap.read()
        if not ret or len(frames_to_select) == 0:
            break
        if frame_ctr not in frames_to_select:
            frame_ctr += 1
            continue
        else:
            frames_to_select.remove(frame_ctr)
            frame_ctr += 1

        bboxes = learner.infer(im)
        if len(bboxes) > 1:
            print('Warning! Multiple faces detected. Using first detected face')

        im = im[int(bboxes[0].top):int(bboxes[0].top+bboxes[0].height),
                int(bboxes[0].left):int(bboxes[0].left+bboxes[0].width), :]

        im = cv2.resize(im, (target_im_size, target_im_size))
        numpy_video.append(im)

    if len(frames_to_select) > 0:
        for i in range(len(frames_to_select)):
            numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))

    numpy_video = np.array(numpy_video)
    return numpy_video


def preprocess_audio(audio_path, sr=22050, target_time=3.6):
    """
    This function preprocesses an audio file. Audio file is cropped/padded to target time.

    Parameters
    ----------
    audio_path : str
        Path to audio file.
    target_time : int, optional
        Target duration of audio. The default is 3.6.
    sr : int, optional
        Sampling rate of audio. The default is 22050.

    Returns
    -------
    y : numpy array
        audio file saved as numpy array.
    """
    y, _ = librosa.core.load(audio_path, sr=sr)
    target_length = int(sr * target_time)
    if len(y) < target_length:
        y = np.array(list(y) + [0 for i in range(target_length - len(y))])
    else:
        remain = len(y) - target_length
        y = y[remain//2:-(remain - remain//2)]
    return y

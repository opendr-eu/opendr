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

import librosa
import numpy as np
import torch as t


def _normalize(tensor: t.Tensor) -> t.Tensor:
    tensor.add_(-tensor.mean())
    tensor.div_(tensor.std())
    minimum = tensor.min()
    maximum = tensor.max()
    if minimum - maximum != 0:
        tensor *= 2
        tensor -= (minimum + maximum)
        tensor /= (maximum - minimum)
    return tensor


def _croppad_to_length(matrix: np.ndarray, length: int):
    matrix = matrix[:, :length]
    if matrix.shape[1] < length:
        matrix = np.hstack((matrix, np.zeros((matrix.shape[0], length - matrix.shape[1]))))
    return matrix


def get_mfcc(values: np.ndarray, sampling_rate: int,
             window_size=0.025, window_stride=0.02,
             n_mfcc=20, normalize=True, length=None):
    fft_window_length = int(sampling_rate * window_size)
    hop_length = int(sampling_rate * window_stride)

    cepstrogram = librosa.feature.mfcc(y=values,
                                       sr=sampling_rate,
                                       n_mfcc=n_mfcc,
                                       hop_length=hop_length,
                                       n_fft=fft_window_length)

    if length is not None:
        cepstrogram = _croppad_to_length(cepstrogram, length)
    cepstrogram = t.FloatTensor(cepstrogram)
    if normalize:
        cepstrogram = _normalize(cepstrogram)
    return cepstrogram

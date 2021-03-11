import librosa
import numpy as np
import torch as t


def _normalize(tensor: t.Tensor) -> t.Tensor:
    tensor.add_(-tensor.mean())
    tensor.div_(tensor.std())
    return tensor


def _croppad_to_length(matrix: np.ndarray, length: int):
    matrix = matrix[:, :length]
    if matrix.shape[1] < length:
        matrix = np.hstack((matrix, np.zeros((matrix.shape[0], length - matrix.shape[1]))))
    return matrix


def get_mfcc(values: np.ndarray, sampling_rate: int,
             window_size=0.04, window_stride=0.03,
             n_mfcc=24, normalize=True, length=None):
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

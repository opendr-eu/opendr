import numpy as np
from scipy.spatial.transform import Rotation


def print_tmat(tmat, note=''):
    return print_sixdof(tmat2sixdof(tmat), note)


def print_array(array, note=''):
    return print_sixdof(array2sixdof(array), note)


def print_sixdof(sixdof, note=''):
    print(string_sixdof(sixdof, note))


def string_tmat(tmat, note=''):
    return string_sixdof(tmat2sixdof(tmat), note)


def string_sixdof(sixdof, note=''):
    string = f'R({np.rad2deg(sixdof["rx"]):>7.3f}, {np.rad2deg(sixdof["ry"]):>7.3f}, ' \
             f'{np.rad2deg(sixdof["rz"]):>7.3f}), T({sixdof["tx"]:>7.3f}, ' \
             f'{sixdof["ty"]:>7.3f}, {sixdof["tz"]:>7.3f}) {note}'
    return string


def create_empty_sixdof():
    sixdof = {'tx': 0, 'ty': 0, 'tz': 0, 'rx': 0, 'ry': 0, 'rz': 0}
    return sixdof


def tmat2sixdof(tmat):
    r = Rotation.from_matrix(tmat[:3, :3]).as_rotvec()
    sixdof = {
        'tx': tmat[0, 3],
        'ty': tmat[1, 3],
        'tz': tmat[2, 3],
        'rx': r[0],
        'ry': r[1],
        'rz': r[2]
    }
    return sixdof


def sixdof2tmat(sixdof):
    tmat = np.eye(4)
    tmat[:3, :3] = Rotation.from_rotvec([sixdof['rx'], sixdof['ry'], sixdof['rz']]).as_matrix()
    tmat[0, 3] = sixdof['tx']
    tmat[1, 3] = sixdof['ty']
    tmat[2, 3] = sixdof['tz']
    return tmat


def tmat2array(tmat):
    sixdof = tmat2sixdof(tmat)
    array = np.zeros((6, 1))
    array[0] = sixdof['rx']
    array[1] = sixdof['ry']
    array[2] = sixdof['rz']
    array[3] = sixdof['tx']
    array[4] = sixdof['ty']
    array[5] = sixdof['tz']
    return array.T.ravel()


def array2tmat(array):
    return sixdof2tmat(array2sixdof(array))


def array2sixdof(array):
    array_ = array.T
    sixdof = {
        'rx': array_[0],
        'ry': array_[1],
        'rz': array_[2],
        'tx': array_[3],
        'ty': array_[4],
        'tz': array_[5]
    }
    return sixdof


def apply_transformation(transformation: np.ndarray, input_data: np.ndarray) -> np.ndarray:
    if len(input_data.shape) != 2 and len(input_data.shape) != 3:
        raise RuntimeError('data should be a 2D or 3D array')
    if len(transformation.shape) != 2:
        raise RuntimeError('transformation should be a 2D array')
    if transformation.shape[0] != transformation.shape[1]:
        raise RuntimeError('transformation should be square matrix')

    if len(input_data.shape) == 2:
        d = input_data.shape[1]
        input_data_ = input_data
    else:
        d = input_data.shape[2]
        input_data_ = input_data.reshape(-1, 3)

    if transformation.shape[0] != d + 1:
        raise RuntimeError('transformation dimension mismatch')
    if np.max(np.abs(transformation[-1, :] - np.r_[np.zeros(d), 1])) > np.finfo(float).eps * 1e3:
        raise RuntimeError('bad transformation')

    x_t = np.c_[input_data_, np.ones((input_data_.shape[0], 1))] @ transformation.T
    x_t = x_t[:, :d].reshape(input_data.shape)

    return x_t
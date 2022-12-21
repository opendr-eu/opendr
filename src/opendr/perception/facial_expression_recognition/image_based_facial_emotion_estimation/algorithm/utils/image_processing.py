
"""
This module implements image processing methods.
Modified based on:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""

import os
import cv2

# Private variables
_MAX_FPS = 30
_FPS = 5
_CAP = None


# Image I/O methods

def set_fps(fps):
    global _FPS
    _FPS = fps


def is_video_capture_open():
    global _CAP

    if _CAP is None:
        return False
    else:
        return _CAP.isOpened()


def initialize_video_capture(source):
    global _CAP

    # If cap is not none, it re-initialize video capture with the new video file
    if not (_CAP is None):
        _CAP.release()
        _CAP = None

    # Read the file
    try:
        _CAP = cv2.VideoCapture(source)
    except Exception as e:
        _CAP = None
        print("Error on trying to read the following file as video: {}".format(source))
        print("Please, check if the file exists, is an image and is not corrupted.")
        print("Supported file format: MPEG-4 (*.mp4).")
        print("Check whether working versions of ffmpeg or gstreamer is installed.")
        raise e

    return not (_CAP is None)


def release_video_capture():
    global _CAP

    try:
        _CAP.release()
    except Exception as e:
        print(e)
    finally:
        _CAP = None

    return _CAP is None


def get_frame():
    """
    Get a frame from a video file.

    :return: (ndarray, float) (Loaded frame, time in seconds).
    """
    global _CAP, _FPS

    to_return_frame = None

    if _CAP is None:
        print("Error on getting frame. cv2.VideoCapture is not initialized.")
    else:
        try:
            if _CAP.isOpened():
                # Skip frames
                for i in range(int(_MAX_FPS / _FPS)):
                    _CAP.grab()

                is_valid_frame, to_return_frame = _CAP.retrieve()

                if not is_valid_frame:
                    to_return_frame = None
        except Exception as e:
            print("Error on getting a frame. Please, double-check if the video file is not corrupted.")
            print("Supported file format: MPEG-4 (*.mp4).")
            print("Check whether working versions of ffmpeg or gstreamer is installed.")
            raise e

    return to_return_frame, (_CAP.get(cv2.CAP_PROP_POS_MSEC) / 1000)


def read(path_to_image, convert_to_grey_scale=False):
    """
    Reads the file as an image.
    :param path_to_image: (string)
    :param convert_to_grey_scale: (bool) opens an image and converts it to a 2d greyscale image.
    :return: (ndarray) 3d (channels last) or 2d image array.
    """

    loaded_image = None
    exception = None

    # Read the file
    try:
        if convert_to_grey_scale:
            loaded_image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
        else:
            loaded_image = cv2.imread(path_to_image, cv2.IMREAD_COLOR)
    except Exception as e:
        loaded_image = None
        exception = e

    # Check if the file has been successfully read as an image
    if loaded_image is None:
        print("Error on trying to read the following file as an image: {}".format(path_to_image))
        print("Please, check if the file exists, is an image and is not corrupted.")
        print("Supported file formats: JPEG (*.jpeg and *.jpg) and Portable Network Graphics (*.png).")

        if exception is None:
            raise RuntimeError("Unable to read the file (unknown error:).")
        else:
            raise exception

    return loaded_image


def write(image, file_path, file_name):
    full_path = os.path.join(file_path, file_name)

    if not os.path.isdir(file_path):
        os.makedirs(file_path)

    cv2.imwrite(full_path, image)

    print("Image successfully saved at: %s" % full_path)


# Color conversion methods

def convert_grey_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def convert_bgr_to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def convert_bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def convert_rgb_to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def convert_rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# Transformation methods

def resize(image, output_size=None, f=None):
    if f is None:
        return cv2.resize(image, output_size)
    else:
        return cv2.resize(image, output_size, fx=f, fy=f)

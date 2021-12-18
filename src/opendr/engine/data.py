# Copyright 2020 Aristotle University of Thessaloniki
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

from pathlib import Path
import cv2
from abc import ABC, abstractmethod
from opendr.engine.target import BoundingBoxList
import numpy as np
import torch
from typing import Union


class Data(ABC):
    """
    Data abstract class allows for representing different types of data. This class serves as the basis for
    more complicated data types. For data classes, conversion from (using the constructor) and to NumPy
    arrays (using the .numpy() method) will be supported to make the library compliant with the standard pipelines
    used by the computer vision and robotics communities.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data):
        self._data = None

    @abstractmethod
    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        pass

    @property
    def data(self):
        """
        Getter of data field.
        This returns the internal representation of the data (which might not be a NumPy array).

        :return: the actual data held by the object
        :rtype: Type of data
        """
        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data. This will perform the necessary type checking (if needed).

        :param: data to be used for creating a vector
        """
        self._data = data

    @abstractmethod
    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        pass


class Vector(Data):
    """
    A class used for representing multidimensional vector data.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data=None):
        super().__init__(data)

        if data is not None:
            self.data = data

    @property
    def data(self):
        """
        Getter of data. Vector class returns a float32 NumPy array.

        :return: the actual data held by the object
        :rtype: A float32 NumPy array
        """
        if self._data is None:
            raise ValueError("Vector is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.

        :param: data to be used for creating a vector
        """
        # Convert input data to a NumPy array
        # Note that will also fail for non-numeric data (which is expected)
        data = np.asarray(data, dtype=np.float32)

        # Check if the supplied vector is 1D
        if len(data.shape) != 1:
            raise ValueError(
                "Only 1-D arrays are supported by Vector. Please supply a data object that can be casted "
                "into a 1-D NumPy array.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return str(self.data)


class Timeseries(Data):
    """
    A class used for representing multidimensional timeseries data.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data=None):
        super().__init__(data)

        if data is not None:
            self.data = data

    @property
    def data(self):
        """
        Getter of data. Vector class returns a float32 NumPy array.

        :return: the actual data held by the object
        :rtype: A float32 NumPy array
        """
        if self._data is None:
            raise ValueError("Timeseries is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.

        :param: data to be used for creating a vector
        """
        # Convert input data to a NumPy array
        # Note that will also fail for non-numeric data (which is expected)
        data = np.asarray(data, dtype=np.float32)

        # Check if the supplied array is 2D
        if len(data.shape) != 2:
            raise ValueError(
                "Only 2-D arrays are supported by Timeseries. Please supply a data object that can be casted "
                "into a 2-D NumPy array. The first dimension corresponds to time and the second to the features.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return str(self.data)


class Image(Data):
    """
    A class used for representing image data.
    OpenDR uses CHW/RGB conventions
    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    - loading an input directly into OpenDR compliant format (open())
    - getting an image into OpenCV-compliant format (opencv()) for visualization purposes
    """

    def __init__(self, data=None, dtype=np.uint8, guess_format=True):
        """
        Image constructor
        :param data: Data to be held by the image object
        :type data: numpy.ndarray
        :param dtype: type of the image data provided
        :type data: numpy.dtype
        :param guess_format: try to automatically guess the type of input data and convert it to OpenDR format
        :type guess_format: bool
        """
        super().__init__(data)

        self.dtype = dtype
        if data is not None:
            # Check if the image is in the correct format
            try:
                data = np.asarray(data)
            except Exception:
                raise ValueError("Image data not understood (cannot be cast to NumPy array).")

            if data.ndim != 3:
                raise ValueError("3-dimensional images are expected")
            if guess_format:
                # If channels are found last and image is a color one, assume OpenCV format
                if data.shape[2] == 3:
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    data = np.transpose(data, (2, 0, 1))
                # If channels are found last and image is not a color one, just perform transpose
                elif data.shape[2] < min(data.shape[0], data.shape[1]):
                    data = np.transpose(data, (2, 0, 1))
            self.data = data
        else:
            raise ValueError("Image is of type None")

    @property
    def data(self):
        """
        Getter of data. Image class returns a *dtype* NumPy array.
        :return: the actual data held by the object
        :rtype: A *dtype* NumPy array
        """
        if self._data is None:
            raise ValueError("Image is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.
        :param: data to be used for creating a vector
        """
        # Convert input data to a NumPy array
        data = np.asarray(data, dtype=self.dtype)

        # Check if the supplied vector is 3D, e.g. (width, height, channels)
        if len(data.shape) != 3:
            raise ValueError(
                "Only 3-D arrays are supported by Image. Please supply a data object that can be casted "
                "into a 3-D NumPy array.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.
        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data.copy()

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.
        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return str(self.data)

    @classmethod
    def open(cls, filename):
        """
        Create an Image from file and return it as RGB.
        :param cls: reference to the Image class
        :type cls: Image
        :param filename: path to the image file
        :type filename: str
        :return: image read from the specified file
        :rtype: Image
        """
        if not Path(filename).exists():
            raise FileNotFoundError('The image file does not exist.')
        data = cv2.imread(filename)
        # Change channel order and convert from HWC to CHW
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        data = np.transpose(data, (2, 0, 1))
        return cls(data)

    def opencv(self):
        """
        Returns the stored image into a format that can be directly used by OpenCV.
        This function is useful due to the discrepancy between the way images are stored:
        HWC/BGR (OpenCV) and CWH/RGB (OpenDR/PyTorch)
        :return: an image into OpenCV compliant-format
        :rtype: NumPy array
        """
        data = np.transpose(self.data, (1, 2, 0))
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        return data

    def convert(self, format='channels_first', channel_order='rgb'):
        """
        Returns the data in channels first/last format using either 'rgb' or 'bgr' ordering.
        :param format: either 'channels_first' or 'channels_last'
        :type format: str
        :param channel_order: either 'rgb' or 'bgr'
        :type channel_order: str
        :return an image (as NumPy array) with the appropriate format
        :rtype NumPy array
        """
        if format == 'channels_last':
            data = np.transpose(self.data, (1, 2, 0))
        elif format == 'channels_first':
            data = self.data.copy()
        else:
            raise ValueError("format not in ('channels_first', 'channels_last')")

        if channel_order == 'bgr':
            # This causes a second copy operation. Can be potentially further optimized in the future.
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        elif channel_order not in ('rgb', 'bgr'):
            raise ValueError("channel_order not in ('rgb', 'bgr')")

        return data


class ImageWithDetections(Image):
    """
    A class used for representing image data with associated 2D object detections.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, image, boundingBoxList: BoundingBoxList, *args, **kwargs):
        super().__init__(image, *args, **kwargs)

        self.boundingBoxList = boundingBoxList

    @property
    def data(self):
        """
        Getter of data. Image class returns a float32 NumPy array.

        :return: the actual data held by the object
        :rtype: A float32 NumPy array
        """
        if self._data is None:
            raise ValueError("Image is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.

        :param: data to be used for creating a vector
        """
        # Convert input data to a NumPy array
        # Note that will also fail for non-numeric data (which is expected)
        data = np.asarray(data, dtype=np.uint8)

        # Check if the supplied vector is 3D, e.g. (width, height, channels)
        if len(data.shape) != 3:
            raise ValueError(
                "Only 3-D arrays are supported by Image. Please supply a data object that can be casted "
                "into a 3-D NumPy array.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return "ImageWithDetections " + str(self.data) + str(self.boundingBoxList)


class Video(Data):
    """
    A class used for representing video data.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data: Union[torch.Tensor, np.ndarray]=None):
        """Construct a new Video

        Args:
            data (Union[torch.Tensor, np.ndarray], optional):
                Video tensor of shape (channels, time_steps, height, width).
                Defaults to None.
        """
        super().__init__(data)

        if data is not None:
            self.data = data

    @property
    def data(self):
        """
        Getter of data. Video class returns a float32 NumPy array.

        :return: the actual data held by the object
        :rtype: A float32 NumPy array
        """
        if self._data is None:
            raise ValueError("Video is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.

        :param: data to be used for creating a vector
        """
        # Convert input data to a NumPy array
        # Note that will also fail for non-numeric data (which is expected)
        data = np.asarray(data, dtype=np.float32)

        # Check if the supplied vector is 4D, e.g. (channels, time, height, width)
        if len(data.shape) != 4:
            raise ValueError(
                "Only 4-D arrays are supported by Image. Please supply a data object that can be casted "
                "into a 4-D NumPy array.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return str(self.data)


class PointCloud(Data):
    """
    A class used for representing point cloud data.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data=None):
        super().__init__(data)

        if data is not None:
            self.data = data

    @property
    def data(self):
        """
        Getter of data. PointCloud class returns a float32 NumPy array.

        :return: the actual data held by the object
        :rtype: A float32 NumPy array in form [length x channels] where channels can be xyz[ref][rgb+]
        """
        if self._data is None:
            raise ValueError("Point Cloud is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.

        :param: data to be used for creating a point cloud
        """
        # Convert input data to a NumPy array
        # Note that will also fail for non-numeric data (which is expected)
        data = np.asarray(data, dtype=np.float32)

        # Check if the supplied array is 2D, e.g. (length, channels)
        if len(data.shape) != 2:
            raise ValueError(
                "Only 2-D arrays are supported by PointCloud. Please supply a data object that can be casted "
                "into a 2-D NumPy array.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return "Points: " + str(self.data)


class PointCloudWithCalibration(PointCloud):
    """
    A class used for representing point cloud data with camera-lidar calibration matrices.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data=None, calib=None, image_shape=None):
        super().__init__(data)

        if data is not None:
            self.data = data

        self.calib = calib
        self.image_shape = image_shape

    @property
    def data(self):
        """
        Getter of data. PointCloudWithCalibration class returns a float32 NumPy array representing a point cloud.

        :return: the actual data held by the object
        :rtype: A float32 NumPy array in form [length x channels] where channels can be xyz[ref][rgb+]
        """
        if self._data is None:
            raise ValueError("Point Cloud is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.

        :param: data to be used for creating a point cloud
        """
        # Convert input data to a NumPy array
        # Note that will also fail for non-numeric data (which is expected)
        data = np.asarray(data, dtype=np.float32)

        # Check if the supplied array is 2D, e.g. (length, channels)
        if len(data.shape) != 2:
            raise ValueError(
                "Only 2-D arrays are supported by PointCloud. Please supply a data object that can be casted "
                "into a 2-D NumPy array.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return "Points: " + str(self.data) + "\nCalib:" + str(self.calib)


class SkeletonSequence(Data):
    """
    A class used for representing a sequence of body skeletons in a video.

    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data=None):
        super().__init__(data)

        if data is not None:
            self.data = data

    @property
    def data(self):
        """
        Getter of data. SkeletonSequence class returns a float32 5-D NumPy array.

        :return: the actual data held by the object
        :rtype: A float32 5-D NumPy array
        """
        if self._data is None:
            raise ValueError("SkeletonSequence is empty")

        return self._data

    @data.setter
    def data(self, data):
        """
        Setter for data.

        :param: data to be used for creating a skeleton sequence
        """
        # Convert input data to a NumPy array
        # Note that will also fail for non-numeric data (which is expected)
        data = np.asarray(data, dtype=np.float32)

        # Check if the supplied vector is 5D, e.g. (num_samples, channels, frames, joints, persons)
        if len(data.shape) != 5:
            raise ValueError(
                "Only 5-D arrays are supported by SkeletonSequence. Please supply a data object that can be casted "
                "into a 5-D NumPy array.")

        self._data = data

    def numpy(self):
        """
        Returns a NumPy-compatible representation of data.

        :return: a NumPy-compatible representation of data
        :rtype: numpy.ndarray
        """
        # Since this class stores the data as NumPy arrays, we can directly return the data
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.

        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return str(self.data)

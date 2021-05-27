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

from abc import ABC, abstractmethod
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
        self.data = data

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
    This class provides abstract methods for:
    - returning a NumPy compatible representation of data (numpy())
    """

    def __init__(self, data=None, dtype=np.uint8):
        super().__init__(data)

        self.dtype = dtype
        if data is not None:
            self.data = data

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
        return self.data

    def __str__(self):
        """
        Returns a human-friendly string-based representation of the data.
        :return: a human-friendly string-based representation of the data
        :rtype: str
        """
        return str(self.data)


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
    A class used for representing point cloud data with camera-lidar calibration matricies.

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

from abc import ABC, abstractmethod
import numpy as np

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

        if data:
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
        
        if data:
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

        # Check if the supplied vector is 1D
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

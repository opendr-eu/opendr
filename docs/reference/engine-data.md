## engine.data Module

The *engine.data* module contains classes representing different types of data.

### Class engine.data.Data
Bases: `abc.ABC`

Data abstract class allows for representing different types of data.
This class serves as the basis for more complicated data types.
For data classes, conversion from (using the constructor) and to [NumPy](https://numpy.org) arrays (using the `numpy()` method) will be supported to make the library compliant with the standard pipelines used by the computer vision and robotics communities.

This class provides abstract methods for returning a [NumPy](https://numpy.org) compatible representation of data `numpy()`.

The [Data](#class-engine.data.data) class has the following public methods:
#### data()
  Return the data argument.
  This method returns the internal representation of the data, which might not be a [NumPy](https://numpy.org) array.

#### numpy()
  Return a [NumPy](https://numpy.org)-compatible representation of data.
  This is an abstract method that returns a `numpy.ndarray` object.

### class engine.data.Timeseries
Bases: `engine.data.Data`

A class used for representing multidimensional timeseries data.

The [Timeseries](#class_engine.data.Timeseries) class has the following public methods:
#### Timeseries(data=None)
  Construct a new [Timeseries](#class_engine.data.Timeseries) object based from *data*.
  *data* is expected to be a 2-D array that can be casted into a 2-D [NumPy](https://numpy.org) array, where the first dimension corresponds to time and the second to the features.

#### data()
  Return *data* argument.
  Return type is float32 [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 2-D array that can be casted into a 2-D [NumPy](https://numpy.org) array, where the first dimension corresponds to time and the second to the features.

#### numpy()
  Return a [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.


### class engine.data.Vector
Bases: `engine.data.Data`

A class used for representing multidimensional vector data.

The [Vector](#class_engine.data.Vector) class has the following public methods:
#### Vector(data=None)
  Construct a new [Vector](#class_engine.data.Vector) object based from *data*.
  *data* is expected to be a 1-D array that can be casted into a 1-D  [NumPy](https://numpy.org) array.

#### data()
  Return *data* argument.
  Return type is float32  [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 1-D array that can be casted into a 1-D  [NumPy](https://numpy.org) array.

#### numpy()
  Return a  [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in  [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.


### class engine.data.Image
Bases: `engine.data.Data`

A class used for representing image data.

The [Image](#class_engine.data.Image) class has the following public methods:
#### Image(data=None, dtype=np.uint8)
  Construct a new [Image](#class_engine.data.Image) object based on *data*.
  *data* is expected to be a 3-D array that can be casted into a 3-D [NumPy](https://numpy.org) array.
  *dtype* is expected to be a [NumPy](https://numpy.org) data type.

#### data()
  Return *data* argument.
  Return type is uint8 [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 3-D array that can be casted into a 3-D [NumPy](https://numpy.org) array, where the
  dimensions can be organized as e.g. (channels, width, height).

#### numpy()
  Return a  [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.


### class engine.data.Video
Bases: `engine.data.Data`

A class used for representing video data.

The [Video](#class_engine.data.Video) class has the following public methods:
#### Video(data=None)
  Construct a new [Video](#class_engine.data.Video) object based on *data*.
  *data* is expected to be a 4-D array of shape (channels, time_steps, height, width).

#### data()
  Return *data* argument.
  Return type is a float32 [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 4-D array that can be casted into a 4-D [NumPy](https://numpy.org) array, where the dimensions can be organized as e.g. (channels, width, height).

#### numpy()
  Return a  [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.


### class engine.data.PointCloud
Bases: `engine.data.Data`

A class used for representing point cloud data.

The [PointCloud](#class_engine.data.PointCloud) class has the following public methods:
#### PointCloud(data=None)
  Construct a new [PointCloud](#class_engine.data.PointCloud) object based on *data*.
  *data* is expected to be a 2-D array that can be casted into a 2-D [NumPy](https://numpy.org) array.

#### data()
  Return *data* argument.
  Return type is float32 [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 2-D array that can be casted into a 2-D [NumPy](https://numpy.org) array, where the
  dimensions can be organized as e.g. (number_of_points, channels).

#### numpy()
  Return a  [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.

### class engine.data.PointCloudWithCalibration
Bases: `engine.data.PointCloud`

A class used for representing point cloud data with a corresponding lidar-camera callibration data.

The [PointCloudWithCalibration](#class_engine.data.PointCloudWithCalibration) class has the following public methods:
#### PointCloudWithCalibration(data=None, calib=None)
  Construct a new [PointCloudWithCalibration](#class_engine.data.PointCloud) object based on *data*.
  *data* is expected to be a 2-D array that can be casted into a 2-D [NumPy](https://numpy.org) array.
  *calib* is expected to be a dictionary with `P`, `R0_rect`, `Tr_velo_to_cam` and `Tr_imu_to_velo` matrices in [NumPy](https://numpy.org)-compatible format.
  - `P[x]` matrices project a point in the rectified referenced camera coordinate to the `camera[x]` image.
  - `R0_rect` is the rectifying rotation for reference coordinate system.
  - `Tr_velo_to_cam` maps a point in point cloud coordinate system to reference coordinate system.
  - `Tr_imu_to_velo` maps a point in IMU coordinate system t0 point cloud coordinate system.

#### data()
  Return *data* argument.
  Return type is float32 [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 2-D array that can be casted into a 2-D [NumPy](https://numpy.org) array, where the
  dimensions can be organized as e.g. (number_of_points, channels).

#### numpy()
  Return a  [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.

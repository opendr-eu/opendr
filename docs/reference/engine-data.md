## engine.data Module

The *engine.data* module contains classes representing different types of data.

### Class engine.data.Data
Bases: `abc.ABC`

[Data](/src/opendr/engine/data.py#L24) abstract class allows for representing different types of data.
This class serves as the basis for more complicated data types.
For data classes, conversion from (using the constructor) and to [NumPy](https://numpy.org) arrays (using the `numpy()` method) will be supported to make the library compliant with the standard pipelines used by the computer vision and robotics communities.

This class provides abstract methods for returning a [NumPy](https://numpy.org) compatible representation of data `numpy()`.

The *Data* class has the following public methods:
#### data()
  Return the data argument.
  This method returns the internal representation of the data, which might not be a [NumPy](https://numpy.org) array.

#### numpy()
  Return a [NumPy](https://numpy.org)-compatible representation of data.
  This is an abstract method that returns a `numpy.ndarray` object.

### class engine.data.Timeseries
Bases: `engine.data.Data`

A class used for representing multidimensional timeseries data.

The [Timeseries](/src/opendr/engine/data.py#L145) class has the following public methods:
#### Timeseries(data=None)
  Construct a new *Timeseries* object based from *data*.
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

The [Vector](/src/opendr/engine/data.py#L79) class has the following public methods:
#### Vector(data=None)
  Construct a new *Vector* object based from *data*.
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

The [Image](/src/opendr/engine/data.py#L211) class has the following public methods:
#### Image(data=None, dtype=np.uint8, guess_format=True)
  Construct a new *Image* object based on *data*.
  *data* is expected to be a 3-D array that can be casted into a 3-D [NumPy](https://numpy.org) array.
  *dtype* is expected to be a [NumPy](https://numpy.org) data type.
  *guess_format* if set to True, then tries to automatically infer whether an [OpenCV](https://opencv.org) image was supplied and then automatically converts it into OpenDR format.
  Note that the OpenDR framework assumes an NCHW/RGB ordering.

#### data()
  Return *data* argument.
  Return type is uint8 [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 3-D array that can be casted into a 3-D [NumPy](https://numpy.org) array, where the
  dimensions can be organized as e.g. (channels, width, height).

#### numpy()
  Return a [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.

#### opencv()
  Return an [OpenCV](https://opencv.org)-compatible representation of data.
  This method transforms the internal CHW/RGB representation into HWC/BGR used by OpenCV.

#### open(filename)
  Construct a new *Image* object from the given image file.

#### convert(format='channels_first', channel_order='rgb')
  Return the data in channels first/last format using either 'rgb' or 'bgr' ordering.
  *format* is expected to be of str type (either 'channels_first' or 'channels_last')
  *channel_order* is expected to be of str type (either 'rgb' or 'bgr')
  Returns an image (as [NumPy](https://numpy.org) array) with the appropriate format
        

### class engine.data.ImageWithDetections
Bases: `engine.data.Image`

A class used for representing image data with a list of detections.
This class is used for methods that rely on an external object detector such as DeepSort for 2D object tracking.

The [ImageWithDetections](/src/opendr/engine/data.py#L358) class has the following public methods:
#### ImageWithDetections(image, boundingBoxList)
  Construct a new *ImageWithDetections* object based on provided data.
  - *image* is expected to be an *Image* or a 3-D array that can be casted into a 3-D [NumPy](https://numpy.org) array.
  - *boundingBoxList* is expected to be a [BoundingBoxList](/src/opendr/engine/target.py#L404).

#### data()
  Return *data* argument.
  Return type is uint8 [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 3-D array that can be casted into a 3-D [NumPy](https://numpy.org) array, where the
  dimensions can be organized as e.g. (channels, width, height).
### class engine.data.Video
Bases: `engine.data.Data`

A class used for representing video data.

The [Video](/src/opendr/engine/data.py#L423) class has the following public methods:
#### Video(data=None)
  Construct a new *Video* object based on *data*.
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

The [PointCloud](/src/opendr/engine/data.py#L496) class has the following public methods:
#### PointCloud(data=None)
  Construct a new *PointCloud* object based on *data*.
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

The [PointCloudWithCalibration](/src/opendr/engine/data.py#L562) class has the following public methods:
#### PointCloudWithCalibration(data=None, calib=None)
  Construct a new *PointCloudWithCalibration* object based on *data*.
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


### class engine.data.SkeletonSequence
Bases: `engine.data.Data`

A class used for representing a sequence of body skeletons in a video.

The [SkeletonSequence](/src/opendr/engine/data.py#L631) class has the following public methods:
#### SkeletonSequence(data=None)
  Construct a new *SkeletonSequence* object based on *data*.
  *data* is expected to be a 5-D array that can be casted into a 5-D [NumPy](https://numpy.org) array.
  The array's dimensions are defined as follows: 
  
  `N, C, T, V, M = array.shape()`,
  
  - `N` is the number of samples, 
  - `C` is the number of channels for each of the body joints 
  - `T` is the number of skeletons in each sequence 
  - `V` is the number of body joints in each skeleton
  - `M` is the number of persons (or skeletons) in each frame. 
  
  Accordingly, an array of size `[10, 3, 300, 18, 2]` contains `10` samples 
  each containing a sequence of `300` skeletons while each skeleton has `2` persons each of which has `18` joints
  and each body joint has `3` channels.  

#### data()
  Return *data* argument.
  Return type is float32 5-D [NumPy](https://numpy.org) array.

#### data(data)
  Set the internal *data* argument.
  *data* is expected to be a 5-D array that can be casted into a 5-D [NumPy](https://numpy.org) array, where the
  dimensions can be organized as e.g. (num_samples, channels, frames, joints, persons).

#### numpy()
  Return a  [NumPy](https://numpy.org)-compatible representation of data.
  Given that *data* argument is already internally stored in [NumPy](https://numpy.org)-compatible format, this method is equivalent to `data()`.

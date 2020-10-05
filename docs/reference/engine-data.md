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

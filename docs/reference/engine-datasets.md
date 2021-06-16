## engine.datasets Module

The *engine.datasets* module contains classes representing different types of datasets that can be used for training and evaluation of models.

### Class engine.datasets.Dataset
Bases: `abc.ABC`

Dataset abstract class for representing different types of datasets.
This class serves as the basis for more complicated datasets.


The [Dataset](#class-engine.datasets.Dataset) class has the following public methods:
#### Dataset()

### Class engine.datasets.DatasetIterator
Bases: `engine.datasets.Dataset`

DatasetIterator serves as an abstraction layer over the different types of datasets.
In this way it provides the opportunity to users to implement different kinds of datasets, while providing a uniform interface.
DatasetIterator should return a tuple of ([engine.data.Data](#class_engine.data.Data), [engine.target.Target](#class_engine.target.Target)) using the index operator `dataset_iterator[idx]`

The [DatasetIterator](#class-engine.datasets.DatasetIterator) class has the following public methods:
#### DatasetIterator()
Construct a new [DatasetIterator](#class-engine.datasets.DatasetIterator) object.

### Examples
* **Creation of a new DatasetIterator class**.  
  ```python
  
  from opendr.engine.data import Image
  from enging.target import Pose
  from opendr.engine.datasets import DatasetIterator

  class SimpleDatasetIterator(DatasetIterator):
    def __init__(self, path, width=1024, height=2605):
      super().__init__()

      self.path = path
      self.width = width
      self.height = height
      self.files = os.listdir(path)

    def __getitem__(self, idx):
      data = np.fromfile(
          str(self.path + "/" + self.files[idx]), dtype=np.uint8, count=-1
      ).reshape([-1, self.width, self.height])

      return (Image(data), Pose([], 0))

    def __len__(self):
      return len(self.files)


  dataset = SimpleDatasetIterator('~/images')
  ```

### Class engine.datasets.ExternalDataset
Bases: `engine.datasets.Dataset`

ExternalDataset provides a way for handling well-known external dataset formats (e.g., COCO, PascalVOC, Imagenet, etc.) directly by OpenDR, without requiring any special effort from the users for writing a specific loader.


The [ExternalDataset](#class-engine.datasets.ExternalDataset) class has the following public methods:
#### ExternalDataset(path, dataset_type)
Construct a new [ExternalDataset](#class-engine.datasets.ExternalDataset) object based on *dataset_type*.
*dataset_type* is expected to be a string with a name of the dataset type like "voc", "coco", "imagenet", "kitti", etc.

#### path()
  Return *path* argument.
  Return type is str.

#### path(value)
  Set the internal *path* argument.
  *path* is expected to be an str.

#### dataset_type()
  Return *dataset_type* argument.
  Return type is str.

#### dataset_type(value)
  Set the internal *dataset_type* argument.
  *dataset_type* is expected to be a string.


### Class engine.datasets.MappedDatasetIterator
Bases: `engine.datasets.DatasetIterator`

MappedDatasetIterator allows to transform elements of the original DatasetIterator.

The [MappedDatasetIterator](#class-engine.datasets.MappedDatasetIterator) class has the following public methods:
#### MappedDatasetIterator(original, map_function)
Construct a new [MappedDatasetIterator](#class-engine.datasets.MappedDatasetIterator) object based on existing *original* [DatasetIterator](#class-engine.datasets.DatasetIterator) and the *map_function*.

### Examples
* **Generation of a MappedDatasetIterator from an existing DatasetIterator**.  
  ```python
  
  from opendr.engine.data import Image
  from enging.target import Pose
  from opendr.engine.datasets import DatasetIterator, MappedDatasetIterator

  class SimpleDatasetIterator(DatasetIterator):
    def __init__(self, path, width=1024, height=2605):
      super().__init__()

      self.path = path
      self.width = width
      self.height = height
      self.files = os.listdir(path)

    def __getitem__(self, idx):
      data = np.fromfile(
          str(self.path + "/" + self.files[idx]), dtype=np.uint8, count=-1
      ).reshape([-1, self.width, self.height])

      return (Image(data), Pose([], 0))

    def __len__(self):
      return len(self.files)


  dataset = SimpleDatasetIterator('~/images')

  # crop 10 pixels from left and from top
  mapped_dataset = MappedDatasetIterator(
      dataset,
      lambda d: (Image(d[0].data()[10:, 10:]), d[1])
  )

  ```

### Class engine.datasets.PointCloudsDatasetIterator
Bases: `engine.datasets.DatasetIterator`

PointCloudsDatasetIterator allows to load point cloud data from disk stored in a [NumPy](https://numpy.org) format.

The [PointCloudsDatasetIterator](#class-engine.datasets.PointCloudsDatasetIterator) class has the following public methods:
#### PointCloudsDatasetIterator(path, num_point_features=4)
  Construct a new [PointCloudsDatasetIterator](#class-engine.datasets.PointCloudsDatasetIterator) object based on path* and *num_point_features*.
  *path* is expected to be a string.
  *num_point_features* is expected to be a number representing the number of features per point.
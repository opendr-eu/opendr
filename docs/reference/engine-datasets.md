## engine.datasets Module

The *engine.datasets* module contains classes representing different types of datasets that can be used for training and evaluation of models.

### Class engine.datasets.Dataset
Bases: `abc.ABC`

Dataset abstract class for representing different types of datasets.
This class serves as the basis for more complicated datasets.


The [Dataset](#class-engine.datasets.Dataset) class has the following public methods:


### Class engine.datasets.DatasetIterator
Bases: `engine.datasets.Dataset`

DatasetIterator serves as an abstraction layer over the different types of datasets.
In this way it provides the opportunity to users to implement different kinds of datasets, while
providing a uniform interface.


The [DatasetIterator](#class-engine.datasets.DatasetIterator) class has the following public methods:

#### __getitem__(idx)
  This method is used for loading the idx-th sample of a dataset along with its annotation.
  Returns a tuple of (`engine.data.Data`, `engine.target.Target`).

#### __len__()
  This method returns the size of the dataset.


### Class engine.datasets.MappedDatasetIterator
Bases: `engine.datasets.DatasetIterator`

MappedDatasetIterator allows to transform elements of the original DatasetIterator

The [MappedDatasetIterator](#class-engine.datasets.MappedDatasetIterator) class has the following public methods:

#### __getitem__(idx)
  This method is used for loading the idx-th sample of a dataset along with its annotation.
  Returns a tuple of (`engine.data.Data`, `engine.target.Target`).

#### __len__()
  This method returns the size of the dataset.


### Class engine.datasets.ExternalDataset
Bases: `engine.datasets.Dataset`

ExternalDataset provides a way for handling well-known external dataset formats
(e.g., COCO, PascalVOC, Imagenet, etc.) directly by OpenDR, without requiring any special effort
from the users for writing a specific loader.


The [ExternalDataset](#class-engine.datasets.ExternalDataset) class has the following public methods:

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
  *dataset_type* is expected to be an str.
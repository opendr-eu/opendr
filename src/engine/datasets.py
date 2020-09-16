from abc import ABC, abstractmethod


class Dataset(ABC):
    """
    Dataset abstract class is used to provide a common ancestor type for dataset classes.
    All classes that provide dataset-related functionality must inherit this class.
    """

    def __init__(self):
        pass


class DatasetIterator(Dataset):
    """
    DatasetIterator serves as an abstraction layer over the different types of datasets.
    In this way it provides the opportunity to users to implement different kinds of datasets, while
    providing a uniform interface.

    This class provides the following abstract methods:
    - __getitem__(i), a getter that allows for retrieving the i-th sample of the dataset, along with its annotation
    - __len__(), which allows for getting the size of the dataset
    """
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, idx):
        """
        This method is used for loading the idx-th sample of a dataset along with its annotation.

        :param idx: the index of the sample to load
        :return: the idx-th sample and its annotation
        :rtype: Tuple of (Data, Target)
        """
        pass

    @abstractmethod
    def __len__(self):
        """
        This method returns the size of the dataset.

        :return: the size of the dataset
        :rtype: int
        """
        pass


class ExternalDataset(Dataset):
    """
    ExternalDataset provides a way for handling well-known external dataset formats
    (e.g., COCO, PascalVOC, Imagenet, etc.) directly by OpenDR, without requiring any special effort
    from the users for writing a specific loader.

    Class attributes:
    - path: path to the external dataset
    - dataset_type:	type of the dataset format (e.g., “voc”, “coco”, “imagenet”, etc.)
    """
    def __init__(self, path, dataset_type):
        super().__init__()
        self.path = path
        self.dataset_type = dataset_type

    @property
    def path(self):
        """
        Getter of path field.
        This returns the path.

        :return: path to the external dataset
        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, value):
        """
        Setter of path field. This will perform the necessary type checking.

        :param value: new path value
        :type value: str
        """
        if type(value) != str:
            raise ValueError('path should be a str')
        else:
            self._path = value

    @property
    def dataset_type(self):
        """
        Getter of dataset_type field.
        This returns the dataset_type.

        :return: type of the dataset format (e.g., “voc”, “coco”, “imagenet”, etc.)
        :rtype: str
        """
        return self._dataset_type

    @dataset_type.setter
    def dataset_type(self, value):
        """
        Setter of dataset_type field. This will perform the necessary type checking.

        :param value: new dataset_type value
        :type value: str
        """
        if type(value) != str:
            raise ValueError('dataset_type should be a str')
        else:
            self._dataset_type = value

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


class BaseLearner(ABC):
    """
    The BaseLearner class contains all shared methods and attributes
    between the learner classes.
    It is extended by Learner, LearnerRL and LearnerActive that add
    extra abstract methods.

    This class provides abstract methods for:
    - saving the model (save())
    - loading the model (load())
    - optimizing the model (optimize())
    - resetting the model's state if required (reset())
    """

    def __init__(self, lr=0.001, iters=10, batch_size=64, optimizer='sgd', lr_schedule='',
                 backbone='default', network_head='', checkpoint_after_iter=0, checkpoint_load_iter=0,
                 temp_path='', device='cuda', threshold=0.0, scale=1.0):

        self._model = None  # Protected attribute; reference to the model object

        # All parameters below are public attributes with appropriate getters/setters
        # Training parameters
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.iters = iters
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.temp_path = temp_path
        self.checkpoint_after_iter = checkpoint_after_iter
        self.checkpoint_load_iter = checkpoint_load_iter
        self.backbone = backbone
        self.network_head = network_head
        self.device = device

        # Inference parameters
        self.threshold = threshold
        self.scale = scale

    @abstractmethod
    def save(self, path):
        """
        Saves the model in the path provided.

        :param path: Path to save directory
        :type path: str
        :return: Whether save succeeded or not
        :rtype: bool
        """
        pass

    @abstractmethod
    def load(self, path):
        """
        Loads a model from the path provided.

        :param path: Path to saved model
        :type path: str
        :return: Whether load succeeded or not
        :rtype: bool
        """
        pass

    @abstractmethod
    def optimize(self, target_device):
        """
        This method optimizes the model based on the parameters provided.

        :param target_device: the optimization's procedure target device
        :type target_device: str
        :return: Whether optimize succeeded or not
        :rtype: bool
        """
        pass

    @abstractmethod
    def reset(self):
        """
        In the case of stateful models, this method can be used to reset
        the model to its initial state.

        :return: None
        :rtype: None
        """
        pass

    @property
    def model(self):
        """
        Getter of _model field.
        This returns the model.

        :return: the neural network model used by this learner
        :rtype: any
        """
        return self._model

    @model.setter
    def model(self, value):
        """
        Setter of _model field.

        :param value: new model value
        :type value: any
        """
        self._model = value

    @property
    def lr(self):
        """
        Getter of learning rate field.
        This returns the value of the learning rate.

        :return: the value of the learning rate
        :rtype: float
        """
        return self._lr

    @lr.setter
    def lr(self, value):
        """
        Setter for learning rate. This will perform the necessary type checking.

        :param value: new lr value
        :type value: float
        """
        if type(value) != float:
            raise TypeError('lr should be a float')
        else:
            self._lr = value

    @property
    def iters(self):
        """
        Getter of iterations field.
        This returns the number of iterations to be run.

        :return: the value of the iterations
        :rtype: int
        """
        return self._iters

    @iters.setter
    def iters(self, value):
        """
        Setter for iterations. This will perform the necessary type checking.

        :param value: new iters value
        :type value: int
        """
        if type(value) != int:
            raise TypeError('iters should be an int')
        else:
            self._iters = value

    @property
    def batch_size(self):
        """
        Getter of batch size field.
        This returns the size of the batch.

        :return: the value of the batch size
        :rtype: int
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """
        Setter for batch size. This will perform the necessary type checking.

        :param value: new batch_size value
        :type value: int
        """
        if type(value) != int:
            raise TypeError('batch_size should be an int')
        else:
            self._batch_size = value

    @property
    def optimizer(self):
        """
        Getter of optimizer field.
        This returns the optimizer to be used during training.

        :return: the name of the optimizer
        :rtype: str
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        """
        Setter for optimizer. This will perform the necessary type checking.

        :param value: new optimizer value
        :type value: str
        """
        if type(value) != str:
            raise TypeError('optimizer should be a str')
        else:
            self._optimizer = value

    @property
    def lr_schedule(self):
        """
        Getter of learning rate scheduler field.
        This returns the lr scheduler to be used during training.

        :return: the name of the lr scheduler
        :rtype: str
        """
        return self._lr_schedule

    @lr_schedule.setter
    def lr_schedule(self, value):
        """
        Setter for lr scheduler. This will perform the necessary type checking.

        :param value: new lr_schedule value
        :type value: str
        """
        if type(value) != str:
            raise TypeError('lr_schedule should be a str')
        else:
            self._lr_schedule = value

    @property
    def backbone(self):
        """
        Getter of backbone field.
        This returns the backbone of the model.

        :return: the backbone of the model
        :rtype: str
        """
        return self._backbone

    @backbone.setter
    def backbone(self, value):
        """
        Setter for backbone architecture. This will perform the necessary type checking.

        :param value: new backbone value
        :type value: str
        """
        if type(value) != str:
            raise TypeError('backbone should be a str')
        else:
            self._backbone = value

    @property
    def network_head(self):
        """
        Getter of network head field.
        This returns the network's head to be used.

        :return: the head of the model
        :rtype: str
        """
        return self._network_head

    @network_head.setter
    def network_head(self, value):
        """
        Setter for model's  head. This will perform the necessary type checking.

        :param value: new network_head value
        :type value: str
        """
        if type(value) != str:
            raise TypeError('network_head should be a str')
        else:
            self._network_head = value

    @property
    def checkpoint_after_iter(self):
        """
        Getter of checkpoint_after_iter field.
        If set to 0, no checkpoints will be created.

        :return: the number of iters between checkpoints
        :rtype: int
        """
        return self._checkpoint_after_iter

    @checkpoint_after_iter.setter
    def checkpoint_after_iter(self, value):
        """
        Setter for checkpoint_after_iter. This will perform the necessary type checking.

        :param value: new checkpoint_after_iter value
        :type value: int
        """
        if type(value) != int:
            raise TypeError('checkpoint_after_iter should be an int')
        else:
            self._checkpoint_after_iter = value

    @property
    def checkpoint_load_iter(self):
        """
        Getter of checkpoint_load_iter field.
        This defines what checkpoint to load based on number of iters.
        If set to 0, no checkpoints will be loaded.

        :return: the number of iters
        :rtype: int
        """
        return self._checkpoint_load_iter

    @checkpoint_load_iter.setter
    def checkpoint_load_iter(self, value):
        """
        Setter for checkpoint_load_iter. This will perform the necessary type checking.

        :param value: new checkpoint_load_iter value
        :type value: int
        """
        if type(value) != int:
            raise TypeError('checkpoint_load_iter should be a int')
        else:
            self._checkpoint_load_iter = value

    @property
    def temp_path(self):
        """
        Getter of temp_path field.
        This returns the path of the temporary directory.

        :return: the path of the temporary directory
        :rtype: str
        """
        return self._temp_path

    @temp_path.setter
    def temp_path(self, value):
        """
        Setter for temp_path. This will perform the necessary type checking.

        :param value: new temp_path value
        :type value: str
        """
        # TODO Maybe check for path validity/existence?
        if type(value) != str:
            raise TypeError('temp_path should be a str')
        else:
            self._temp_path = value

    @property
    def device(self):
        """
        Getter of device field.
        This returns the value of the device.

        :return: the value of the device
        :rtype: str
        """
        return self._device

    @device.setter
    def device(self, value):
        """
        Setter for device. This will perform the necessary type checking.

        :param value: new device value
        :type value: str
        """
        if type(value) != str:
            raise TypeError('device should be a str')
        else:
            self._device = value

    @property
    def threshold(self):
        """
        Getter of threshold field. Returns the threshold field value.
        If set to a value different from 0, then allows for using a custom detection/recognition threshold.

        :return: The threshold value
        :rtype: float
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        """
        Setter for threshold. This will perform the necessary type checking.

        :param value: New threshold value
        :type value: float
        """
        if type(value) != float:
            raise TypeError('threshold should be a float')
        else:
            self._threshold = value

    @property
    def scale(self):
        """
        Allows for scaling down the input (values < 1), to accelerate the inference process, or to perform
        analysis on higher resolution than the default (values > 1).

        :return: The scale value
        :rtype: float
        """
        return self._scale

    @scale.setter
    def scale(self, value):
        """
        Setter for scale. This will perform the necessary type checking.

        :param value: New scale value
        :type value: float
        """
        if type(value) != float:
            raise TypeError('scale should be a float')
        else:
            self._scale = value

    @staticmethod
    def get_hyperparameters():
        """
        Static method for obtaining the names of hyperparameters that can be tuned, their type and their possible
        values. This method allows to easily perform hyperparameter tuning.

        :return: Description of hyperparameters.
        :rtype: list[dict[str, any]]
        """
        return None

    @staticmethod
    def get_objective_function():
        """
        Static method for obtaining a mapping from the output of the eval method to a scalar objective that is to be
        minimized during hyperparameter tuning. This method allows to easily perform hyperparameter tuning.

        :return: Objective function that maps the output from the eval_method to an objective scalar value.
        :rtype: callable[any]
        """
        return None


class Learner(BaseLearner):
    """
    The Learner abstract class can be used to implement most perception algorithms.

    It extends the BaseLearner class by adding methods whose implementations
    should be specific for non-RL, non-active algorithms.

    All classes responsible for implementing any perception algorithm should
    inherit and implement the abstract Learner class to ensure that a common
    interface will be provided for training.

    This class extends BaseLearner by adding the following abstract methods for
    non-RL, non-active algorithms:
    - training algorithms (fit())
    - evaluating the performance of a trained model (eval())
    - performing inference (infer())
    """

    def __init__(self, lr=0.001, iters=10, batch_size=64, optimizer='sgd', lr_schedule='',
                 backbone='default', network_head='', checkpoint_after_iter=0, checkpoint_load_iter=0,
                 temp_path='', device='cuda', threshold=0.0, scale=1.0):
        super(Learner, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer,
                                      lr_schedule=lr_schedule, backbone=backbone, network_head=network_head,
                                      checkpoint_after_iter=checkpoint_after_iter,
                                      checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                      device=device, threshold=threshold, scale=scale)

    @abstractmethod
    def fit(self, dataset, val_dataset=None, logging_path='', silent=True, verbose=True):
        """
        This method is used for training the algorithm on a train dataset and
        validating on a val dataset.

        Can be parameterized based on the learner attributes and custom hyperparameters
        added by the implementation and returns stats regarding training and validation.

        :param dataset: Object that holds the training dataset
        :type dataset: Dataset class type
        :param val_dataset: Object that holds the validation dataset
        :type val_dataset: Dataset class type, optional
        :param verbose: if set to True, enables the maximum logging verbosity (depends on the actual algorithm)
        :type verbose: bool, optional
        :param silent: if set to True, disables printing training progress reports to STDOUT
        :type silent: bool, optional
        :param logging_path: path to save tensorboard log files. If set to None or ‘’, tensorboard logging is disabled
        :type logging_path: str, optional
        :return: Returns stats regarding training and validation
        :rtype: dict
        """
        pass

    @abstractmethod
    def eval(self, dataset):
        """
        This method is used to evaluate the algorithm on a dataset
        and returns stats regarding the evaluation ran.

        :param dataset: Object that holds the dataset to evaluate the algorithm on
        :type dataset: Dataset class type
        :return: Returns stats regarding evaluation
        :rtype: dict
        """
        pass

    @abstractmethod
    def infer(self, batch):
        """
        This method performs inference on the batch provided.

        Can be parameterized based on the learner attributes and custom hyperparameters
        added by the implementation and returns a list of predicted targets.

        :param batch: Object that holds a batch of data to run inference on
        :type batch: Data class type
        :return: A list of predicted targets
        :rtype: list of Target class type objects
        """
        pass


class LearnerRL(BaseLearner):
    """
    The LearnerRL abstract class can be used to implement algorithms trained
    using RL.

    It extends the BaseLearner class by adding methods whose implementations
    should be specific for RL algorithms.

    This class extends BaseLearner by adding the following abstract methods for
    RL algorithms:
    - training algorithms (fit())
    - evaluating the performance of a trained model (eval())
    - performing inference (infer())
    """

    def __init__(self, lr=0.001, iters=10, batch_size=64, optimizer='sgd', lr_schedule='',
                 backbone='default', network_head='', checkpoint_after_iter=0, checkpoint_load_iter=0,
                 temp_path='', device='cuda', threshold=0.0, scale=1.0):
        super(LearnerRL, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer,
                                        lr_schedule=lr_schedule, backbone=backbone, network_head=network_head,
                                        checkpoint_after_iter=checkpoint_after_iter,
                                        checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                        device=device, threshold=threshold, scale=scale)

    @abstractmethod
    def fit(self, env, val_env=None, logging_path='', silent=True, verbose=True):
        """
        This method is used for training the RL algorithm on a
        training environment and validating on a validation environment.

        Can be parameterized based on the learner attributes and custom hyperparameters
        added by the implementation and returns stats regarding training and validation.

        :param env: Object that holds the training environment
        :type env: BaseEnv class type
        :param val_env: Object that holds the validation environment
        :type val_env: BaseEnv class type
        :param verbose: if set to True, enables the maximum logging verbosity (depends on the actual algorithm)
        :type verbose: bool, optional
        :param silent: if set to True, disables printing training progress reports to STDOUT
        :type silent: bool, optional
        :param logging_path: path to save tensorboard log files. If set to None or ‘’, tensorboard logging is disabled
        :type logging_path: str, optional
        :return: Returns stats regarding training and validation
        :rtype: dict
        """
        pass

    @abstractmethod
    def eval(self, env):
        """
        This method is used to evaluate the algorithm on an environment
        and returns stats regarding the evaluation ran.

        :param env: Object that holds the environment to evaluate the algorithm on
        :type env: BaseEnv class type
        :return: Returns stats regarding evaluation
        :rtype: dict
        """
        pass

    @abstractmethod
    def infer(self, batch):
        """
        This method performs inference on the batch provided.

        Can be parameterized based on the learner attributes and custom hyperparameters
        added by the implementation and returns a list of predicted RL targets.

        :param batch: Object that holds a batch of data to run inference on
        :type batch: Data class type
        :return: A list of predicted RL targets
        :rtype: list of RLTarget class type objects
        """
        pass


class LearnerActive(BaseLearner):
    """
    The LearnerActive abstract class can be used to implement active algorithms.

    It extends the BaseLearner class by adding methods whose implementations
    should be specific for active algorithms.

    This class extends BaseLearner by adding the following abstract methods for
    active algorithms:
    - training algorithms (fit())
    - evaluating the performance of a trained model (eval())
    - performing inference (infer())
    """

    def __init__(self, lr=0.001, iters=10, batch_size=64, optimizer='sgd', lr_schedule='',
                 backbone='default', network_head='', checkpoint_after_iter=0, checkpoint_load_iter=0,
                 temp_path='', device='cuda', threshold=0.0, scale=1.0):
        super(LearnerActive, self).__init__(lr=lr, iters=iters, batch_size=batch_size, optimizer=optimizer,
                                            lr_schedule=lr_schedule, backbone=backbone, network_head=network_head,
                                            checkpoint_after_iter=checkpoint_after_iter,
                                            checkpoint_load_iter=checkpoint_load_iter, temp_path=temp_path,
                                            device=device, threshold=threshold, scale=scale)

    @abstractmethod
    def fit(self, dataset, env, val_dataset=None, val_env=None, logging_path='', silent=True, verbose=True):
        """
        This method is used for training the algorithm on a training
        environment/dataset and validating on a validation environment/dataset.

        Can be parameterized based on the algorithm with the hyperparameters
        argument and returns stats regarding training and validation.

        :param dataset: Object that holds the training dataset
        :type dataset: Dataset class type
        :param env: Object that holds the training environment
        :type env: BaseEnv class type
        :param val_dataset: Object that holds the validation dataset
        :type val_dataset: Dataset class type
        :param val_env: Object that holds the validation environment
        :type val_env: BaseEnv class type
        :param verbose: if set to True, enables the maximum logging verbosity (depends on the actual algorithm)
        :type verbose: bool, optional
        :param silent: if set to True, disables printing training progress reports to STDOUT
        :type silent: bool, optional
        :param logging_path: path to save tensorboard log files. If set to None or ‘’, tensorboard logging is disabled
        :type logging_path: str, optional
        :return: Returns stats regarding training and validation
        :rtype: dict
        """
        pass

    @abstractmethod
    def eval(self, dataset, env):
        """
        This method is used to evaluate the algorithm on an environment/dataset
        and returns stats regarding the evaluation ran.

        :param dataset: Object that holds the dataset to evaluate the algorithm on
        :type dataset: Dataset class type
        :param env: Object that holds the environment to evaluate the algorithm on
        :type env: BaseEnv class type
        :return: Returns stats regarding evaluation
        :rtype: dict
        """
        pass

    @abstractmethod
    def infer(self, batch):
        """
        This method performs inference on the batch provided.

        Can be parameterized based on the learner attributes and custom hyperparameters
        added by the implementation and returns a list of predicted targets.

        :param batch: Object that holds a batch of data to run inference on
        :type batch: Data class type
        :return: A list of predicted targets
        :rtype: list of Target class type objects
        """
        pass

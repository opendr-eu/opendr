# Copyright 2020-2021 OpenDR European Project
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

import inspect
import optuna
from optuna.study.study import Study
from optuna.trial import Trial
from abc import ABCMeta
from copy import deepcopy
from typing import Optional, Union, Dict, Any, List, Callable, Type
from tabulate import tabulate
from opendr.engine.learners import Learner, LearnerRL, LearnerActive


class HyperparameterTuner(object):
    """HyperparameterTuner

    This tool can be used to perform hyperparameter tuning for any of the learner classes available in the OpenDR
    toolkit.
    """
    def __init__(
            self,
            learner_class: Union[Type[Learner], Type[LearnerRL], Type[LearnerActive]],
            study: Optional[Study] = None,
    ) -> None:
        """Constructor of the HyperparameterTuner.

        Example:
            import optuna
            from opendr.utils.hyperparameter_tuner import HyperparameterTuner
            from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner

            study = optuna.create_study(study_name="detr_study", load_if_exists=True)
            tuner = HyperparameterTuner(DetrLearner, study=study)

        :param learner_class: OpenDR learner class for which hyperparameters should be tuned.
        :type learner_class: Union[Type[Learner], Type[LearnerRL], Type[LearnerActive]]
        :param study: "A study corresponds to an optimization task, i.e., a set of trials." taken from
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study
        If not provided, a Study object will be created with the default parameters, which can be found here:
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study
        :type study: Optional[Study]
        """
        # If the user does not provide a Optuna Study, a default one is created
        if study is None:
            print('No Study object is provided, a default one will be created using optuna.create_study().')
            study = optuna.create_study()
        self._study = study

        assert type(learner_class) is ABCMeta, "The learner_class should be an uninitialized class object."
        self._learner_class = learner_class
        self.init_arguments = None
        self.fit_arguments = None
        self.eval_arguments = None
        self.iters = None
        self.iters_is_optimized = False
        self.hyperparameters = self._learner_class.get_hyperparameters()
        self._objective_function = self._learner_class.get_objective_function()

        # Check if learner class has an implementation of get_hyperparameters()
        if self.hyperparameters is None:
            print(
                'The get_hyperparameters method is not implemented in the {} class.\n'.format(learner_class.__name__) +
                'The optimize method can not be run without specifying the hyperparameters argument.'
            )
        else:
            print(
                'The get_hyperparameters method is implemented in the {} class.\n'.format(learner_class.__name__) +
                'The optimize method can be run without specifying the hyperparameters argument.\n' +
                'If the hyperparameters argument is not specified for the optimize method, the following ' +
                'hyperparameters will be tuned:'
            )
            print(tabulate(self.hyperparameters, headers='keys'))
        # Check if learner class has an implementation of get_objective_function()
        if self._objective_function is None:
            print(
                'The get_objective_function method is not implemented in the {} class.\n'.format(
                    learner_class.__name__) +
                'The optimize method can not be run without specifying the hyperparameters argument.`'
            )
        else:
            print(
                'The get_hyperparameters method is implemented in the {} class.\n'.format(learner_class.__name__) +
                'The optimize method can be run without specifying the objective_function argument.'
            )

    def optimize(
            self,
            hyperparameters: Optional[List[Dict[str, Any]]]=None,
            init_arguments: Optional[Dict[str, Any]]=None,
            fit_arguments: Optional[Dict[str, Any]]=None,
            eval_arguments: Optional[Dict[str, Any]]=None,
            objective_function: Optional[Callable]=None,
            n_trials: Optional[int]=None,
            timeout: Optional[float]=None,
            n_jobs: Optional[int]=1,
            show_progress_bar: Optional[bool]=False,
            verbose: Optional[bool]=False,
    ) -> Dict[str, Any]:
        """Hyperparameter tuning using Optuna.

        Example:
            from opendr.utils.hyperparameter_tuner.hyperparameter_tuner import HyperparameterTuner
            from opendr.perception.object_detection_2d.detr.detr_learner import DetrLearner
            from opendr.engine.datasets import ExternalDataset

            # Create a coco dataset, containing training and evaluation data
            dataset = ExternalDataset(path='./coco', dataset_type='COCO')

            # Specify the hyperparameters that we want to tune
            hyperparameters = [
                {'name': 'optimizer', 'type': 'categorical', 'choices': ['sgd', 'adam']},
                {'name': 'lr', 'type': 'float', 'low': 0.00001, 'high': 0.01, 'log': True},
            ]

            # Specify the arguments that are required for the fit method
            fit_arguments = {'dataset': dataset}

            # Specify the arguments that are required for the eval method
            eval_arguments = {'dataset': dataset}

            # Define an objective function that we wish to minimize
            def objective_function(eval_stats):
                return eval_stats['loss']

            # Specify timeout such that optimization is performed for 2 hours
            timeout = 7200

            # Initialize the tuner
            tuner = HyperparameterTuner(DetrLearner)

            # Optimize
            best_parameters = tuner.optimize(
                hyperparameters=hyperparameters,
                fit_arguments=fit_arguments,
                eval_arguments=eval_arguments,
                objective_function=objective_function,
                timeout=timeout,
            )

            # Initialize learner with the tuned hyperparameters
            learner = DetrLearner(**best_parameters)

        :param hyperparameters: Specifies which hyperparameters should be tuned that are set during initialization of
        the learner. The *hyperparameters* argument should be a list of dictionaries, where each dictionary describes a
        hyperparameter. Required keys in these dictionaries are 'name' and 'type', where the value for 'name' should
        correspond to an argument name of the learner's constructor. Value for 'type' should be in
        ['categorial', 'discrete_uniform', 'float', 'int', 'loguniform', 'uniform']. Furthermore, the required  and
        optional (*[] are optional) keys for each type are the following:
            categorical:            choices
            discrete_uniform:       low, high, q
            float:                  low, high, *[, step, log]
            int:                    low, high,
            loguniform:             low, high
            uniform:                low, high
        If not specified, the hyperparameters will be obtained from the learner_class.get_hyperparameters() method.
        If this method is not implemented and the hyperparameters are not specified, hyperparameter tuning cannot be
        performed and an error is raised.
        :type hyperparameters: List[Dict[str, Any]]
        :param init_arguments: Specifies the arguments that are required for initializing the learner. Together with
        the *hyperparameters*, they define the arguments for constructing the learner. The *init_arguments* argument
        should be a dictionary, where each key corresponds to an argument name of the learner's constructor. During
        optimization, the learner will be constructed with the value that corresponds to the key.
        :type init_arguments: Dict[str, Any]
        :param fit_arguments: Specifies the arguments that are required for calling the fit method. The *fit_arguments*
        argument should be a dictionary, where each key corresponds to an argument name of the learner's fit method.
        During optimization, the fit method will be called with the value that corresponds to the key.
        :type fit_arguments: Dict[str, Any]
        :param eval_arguments: Specifies the arguments that are required for calling the eval method. The
        *eval_arguments* argument should be a dictionary, where each key corresponds to an argument name of the
        learner's eval method. During optimization, the eval method will be called with the value that corresponds to
        the key.
        :type eval_arguments: Dict[str, Any]
        :param objective_function: Function that maps the output from the eval method to a scalar objective value. The
        optimal hyperparameters should correspond to a minimum of the objective_function. The input of this callable
        should be the output of the learner's eval method. If not specified, the objective_function will be obtained
        from the learner_class.get_objective_function() method. If this method is not implemented in the learner_class
        and the objective_function is not specified, hyperparameter tuning cannot be performed and an error is raised.
        :type objective_function: Callable
        :param n_trials: "The number of trials. If this argument is set to None, there is no limitation on the number of
        trials. If timeout is also set to None, the study continues to create trials until it receives a termination
        signal such as Ctrl+C or SIGTERM." taken from
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        :type n_trials: int
        :param timeout: "Stop study after the given number of second(s). If this argument is set to None, the study is
        executed without time limitation. If n_trials is also set to None, the study continues to create trials until it
        receives a termination signal such as Ctrl+C or SIGTERM." taken from
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        :type timeout: float
        :param n_jobs: "The number of parallel jobs. If this argument is set to -1, the number is set to CPU count."
        taken from
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        :type n_jobs: int
        :param show_progress_bar: "Flag to show progress bars or not. To disable progress bar, set this False.
        Currently, progress bar is experimental feature and disabled when n_jobs ≠1." taken from
        https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
        :type show_progress_bar: bool
        :param verbose: Set verbosity level.
        :type verbose: bool
        :return: Dictionary with optimal hyperparameters and provided init_arguments, such that the learner can be
        initialized directly with the returned dict.
        :rtype: Dict[str, Any]
        """
        if hyperparameters is not None:
            self.hyperparameters = hyperparameters
        else:
            assert (self.hyperparameters is not None), \
                "Cannot tune hyperparameters, since hyperparameters are not defined."
        if init_arguments is not None:
            self.init_arguments = init_arguments
        else:
            self.init_arguments = {}
        if fit_arguments is not None:
            self.fit_arguments = fit_arguments
        else:
            self.fit_arguments = {}
        if eval_arguments is not None:
            self.eval_arguments = eval_arguments
        else:
            self.eval_arguments = {}
        if objective_function is not None:
            self._objective_function = objective_function
        else:
            assert (self._objective_function is not None), \
                "Cannot tune hyperparameters, since objective_function is not defined."

        # Check validity of hyperparameter definitions
        self._check_hyperparameters()

        # Set the number of iterations
        self._set_iters()

        # Set verbosity
        if verbose:
            verbosity = optuna.logging.INFO
        else:
            verbosity = optuna.logging.WARN
        optuna.logging.set_verbosity(verbosity)

        # Optimize
        self._study.optimize(
            self._objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=show_progress_bar,
        )

        # Get hyperparameters from trial with lowest objective value
        best_trial = self._study.best_trial

        # Reset the number of iters to the value that was specified by the user
        if not self.iters_is_optimized and 'iters' in self.init_arguments.keys():
            self.init_arguments['iters'] = self.iters
        return {**self.init_arguments, **best_trial.params}

    @property
    def study(self) -> Study:
        """Getter of _study field.

        This returns the study.

        :return: the Optuna study
        :rtype: optuna.study.study.Study
        """
        return self._study

    def _check_hyperparameters(self) -> None:
        """Check the validity of the provided hyperparameters argument.
        :return: None
        """
        required_args = {
            'categorical': ['choices'],
            'discrete_uniform': ['low', 'high', 'q'],
            'float': ['low', 'high'],
            'int': ['low', 'high'],
            'loguniform': ['low', 'high'],
            'uniform': ['low', 'high'],
        }
        optional_args = {
            'categorical': [],
            'discrete_uniform': [],
            'float': ['step', 'log'],
            'int': ['step', 'log'],
            'loguniform': [],
            'uniform': [],
        }
        for hyperparameter in self.hyperparameters:
            valid = False
            assert 'name' in hyperparameter, "{} does not contain the required key 'name'".format(hyperparameter)
            assert 'type' in hyperparameter, "{} does not contain the required key 'type'".format(hyperparameter)
            type = hyperparameter['type']
            for key, value in required_args.items():
                if type.lower() == key:
                    for required_argument in value:
                        assert required_argument in hyperparameter, \
                            "{} does not contain required key {}".format(hyperparameter, required_argument)
                    for hyperparameter_key in hyperparameter.keys():
                        valid_key = (
                            hyperparameter_key in ['type', 'name'] or
                            hyperparameter_key in required_args[type] or
                            hyperparameter_key in optional_args[type]
                        )
                        assert valid_key, "{} is not a valid argument for type {}".format(hyperparameter_key, type)
                    valid = True
                    break
            assert valid, \
                "{} is an invalid hyperparameter type, supported types are: {}".format(type, required_args.keys())

    def _suggest_hyperparameters(self, trial: Trial) -> Dict[str, any]:
        """Get suggested hyperparameters from Optuna trial.

        :param trial: The Optuna trial.
        :type trial: optuna.trial.Trial
        :return: Suggested hyperparameters
        :rtype: Dict[str, any]
        """
        suggested_hyperparameters = {}
        suggest_functions = {
            'categorical': trial.suggest_categorical,
            'discrete_uniform': trial.suggest_discrete_uniform,
            'float': trial.suggest_float,
            'int': trial.suggest_int,
            'loguniform': trial.suggest_loguniform,
            'uniform': trial.suggest_uniform,
        }
        hyperparameters = deepcopy(self.hyperparameters)
        for hyperparameter in hyperparameters:
            name = hyperparameter['name']
            type = hyperparameter['type']
            hyperparameter.pop('type')
            suggested_hyperparameters[name] = suggest_functions[type](**hyperparameter)
        return suggested_hyperparameters

    def _objective(self, trial: Trial) -> float:
        """Objective function that optuna tries to minimize.

        :param trial: The Optuna trial.
        :type trial: optuna.trial.Trial
        :return: Scalar objective value.
        :rtype: float
        """
        hyperparameters = self._suggest_hyperparameters(trial)
        if self.iters_is_optimized:
            self.iters = deepcopy(hyperparameters['iters'])
            hyperparameters['iters'] = 1
        learner = self._learner_class(**{**self.init_arguments, **hyperparameters})

        for iter in range(self.iters):
            learner.fit(**self.fit_arguments)
            eval_stats = learner.eval(**self.eval_arguments)
            objective_value = self._objective_function(eval_stats)
            trial.report(objective_value, iter)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return objective_value

    def _set_iters(self) -> None:
        """Set the number of iterations.

        We set iters to 1 in the learner and call the fit and eval method for the number of iters instead. This allows
        to prune based on the intermediate result.

        :return: None
        """

        for idx, hyperparameter in enumerate(self.hyperparameters):
            # We first check if iters is a hyperparameter that is to be tuned
            if hyperparameter['name'] == 'iters':
                self.iters_is_optimized = True

        if not self.iters_is_optimized:
            if 'iters' in self.init_arguments.keys():
                # Check if iters is in the prescribed init arguments
                self.iters = deepcopy(self.init_arguments['iters'])
                self.init_arguments['iters'] = 1
            else:
                # Next we check if iters is an argument of the constructor of the learner
                argspec = inspect.getfullargspec(self._learner_class.__init__)
                if 'iters' in argspec.args:
                    self.iters = argspec.defaults[argspec.args.index('iters') + 1]
                    self.init_arguments['iters'] = 1
                else:
                    # iters is not an argument of the learner
                    self.iters = 1

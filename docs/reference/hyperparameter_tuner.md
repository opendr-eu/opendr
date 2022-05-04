## hyperparameter_tuner module

The *hyperparameter_tuner* module contains the *HyperparameterTuner* class.

### Class HyperparameterTuner
Bases: `object`

The *HyperparameterTuner* class is a wrapper around the [Optuna](https://optuna.org/) [[1]](#optuna-paper) hyperparameter tuning framework.
It allows to easily perform hyperparameter tuning with any of the learners that are present in the OpenDR toolkit.

The [HyperparameterTuner](../../src/opendr/utils/hyperparameter_tuner/hyperparameter_tuner.py) class has the
following public methods:

#### `HyperparameterTuner` constructor
```python
HyperparameterTuner(self, learner_class, study)
```

Constructor parameters:

- **learner_class**: *Union[Type[Learner], Type[LearnerRL], Type[LearnerActive]]*\
  OpenDR learner class for which hyperparameters should be tuned.
  Note that this learner should not be initialized, e.g. *learner_class* can be *detr_learner* but not *detr_learner()*.
- **study**: *optuna.study.study.Study, default=None*\
  "A study corresponds to an optimization task, i.e., a set of trials" (taken from [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.)).
  If not provided, a Study object will be created with the default parameters, which can be found [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html#optuna.study.create_study).

#### `HyperparameterTuner.optimize`
```python
HyperparameterTuner.optimize(self, hyperparameters, init_arguments, fit_arguments, eval_arguments, objective_function, n_trials, timeout, n_jobs, show_progress_bar, verbose)
```

This method allows to perform hyperparameter tuning with Optuna.

Parameters:

- **hyperparameters**: *List[Dict[str, Any]], default=None*\
  Specifies which hyperparameters should be tuned that are set during initialization of the learner.
  The *hyperparameters* argument should be a list of dictionaries, where each dictionary describes a hyperparameter.
  Required keys in these dictionaries are *'name'* and *'type'*, where the value for 'name' should correspond to an
  argument name of the learner's constructor.
  Value for *'type'* should be in *['categorial', 'discrete_uniform', 'float', 'int', 'loguniform', 'uniform']*.
  Furthermore, the required  and optional keys for each type are the following:

    | **Type**               | **Required** | **Optional** |
    | ---------------------- | -------------| ------------ |
    | categorical            | choices      | -            |  
    | discrete_uniform       | low, high, q | -            |
    | float                  | low, high    | step, log    |
    | int                    | low, high    | -            |
    | loguniform             | low, high    | -            |
    | uniform                | low, high    | -            |

  More information on these parameters can be found [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial).
  If not specified, the hyperparameters will be obtained from the *learner_class*.get_hyperparameters() method.
  If this method is not implemented in the *learner_class* and *hyperparameters* are not specified, hyperparameter
  tuning cannot be performed and an error is raised.
- **init_arguments**: *Dict[str, Any], default=None*\
  Specifies the arguments that are required for initializing the learner.
  Together with the *hyperparameters*, they define the arguments for constructing the learner.
  The *init_arguments* argument should be a dictionary, where each key corresponds to an argument name of the learner's
  constructor.
  During optimization, the learner will be constructed with the value that corresponds to the key.
  If not provided, it will be assumed to be an empty dict.
- **fit_arguments**: *Dict[str, Any], default=None*\
  Specifies the arguments that are required for calling the fit method.
  The *fit_arguments* argument should be a dictionary, where each key corresponds to an argument name of the learner's
  fit method.
  During optimization, the fit method will be called with the value that corresponds to the key.
  If not provided, it will be assumed to be an empty dict.  
- **eval_arguments**: *Dict[str, Any], default=None*\
  Specifies the arguments that are required for calling the eval method.
  The *eval_arguments* argument should be a dictionary, where each key corresponds to an argument name of the learner's
  eval method.
  During optimization, the eval method will be called with the value that corresponds to the key.
  If not provided, it will be assumed to be an empty dict.  
- **objective_function**: *Callable, default=None*\
  Function that maps the output from the eval method to a scalar objective value.
  The optimal hyperparameters should correspond to a minimum of the objective_function.
  The input of this callable should be the output of the learner's eval method.
  If not specified, the objective_function will be obtained from the *learner_class*.get_objective_function() method.
  If this method is not implemented in the *learner_class* and the *objective_function* is not specified, hyperparameter
  tuning cannot be performed and an error is raised.
- **n_trials**: *int, default=None*\
  "The number of trials. If this argument is set to None, there is no limitation on the number of trials.
  If timeout is also set to None, the study continues to create trials until it receives a termination signal such as
  Ctrl+C or SIGTERM", taken from [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).
- **timeout**: *float, default=None*\
  "Stop study after the given number of second(s).
  If this argument is set to None, the study is executed without time limitation.
  If n_trials is also set to None, the study continues to create trials until it receives a termination signal such as
  Ctrl+C or SIGTERM", taken from [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).
- **n_jobs**: *int, default=None*\
  "The number of parallel jobs. If this argument is set to -1, the number is set to CPU count", taken from [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).
- **show_progress_bar**: *bool, default=False*\
  "Flag to show progress bars or not. To disable progress bar, set this False.
  Currently, progress bar is experimental feature and disabled when n_jobs ≠1", taken from [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).
- **verbose**: *bool, default=False*\
  If *True*, maximum verbosity is enabled.

#### Demos and tutorial

A demo showcasing the usage and functionality of the *HyperparameterTuner* is available
[here](../../projects/utils/hyperparameter_tuner/hyperparameter_tuner_demo.py).
Also, a tutorial in the form of a Jupyter Notebook is available
[here](../../projects/utils/hyperparameter_tuner/hyperparameter_tuning_tutorial.ipynb).


#### Examples

* **Hyperparameter tuning example with the [DetrLearner](detr.md).**

  This example shows how to tune hyperparameters of the *DetrLearner*.

  ```python
  from opendr.utils.hyperparameter_tuner import HyperparameterTuner
  from opendr.perception.object_detection_2d import DetrLearner
  from opendr.engine.datasets import ExternalDataset

  # Create a coco dataset, containing training and evaluation data
  dataset = ExternalDataset(path='./my_dataset', dataset_type='COCO')

  # Specify the arguments that are required for the fit method
  fit_arguments = {'dataset': dataset}

  # Specify the arguments that are required for the eval method
  eval_arguments = {'dataset': dataset}

  # Specify timeout such that optimization is performed for 4 hours
  timeout = 14400

  # Initialize the tuner
  tuner = HyperparameterTuner(DetrLearner)

  # Optimize
  best_parameters = tuner.optimize(
    fit_arguments=fit_arguments,
    eval_arguments=eval_arguments,
    timeout=timeout,
  )

  # Initialize learner with the tuned hyperparameters
  learner = DetrLearner(**best_parameters)
  ```


* **Custom hyperparameter tuning example with the [DetrLearner](detr.md)**

  This example shows how to tune a selection of the hyperparameters of the *DetrLearner* and
  how to specify an objective function.

  ```python
  from opendr.utils.hyperparameter_tuner import HyperparameterTuner
  from opendr.perception.object_detection_2d import DetrLearner
  from opendr.engine.datasets import ExternalDataset

  # Create a coco dataset, containing training and evaluation data
  dataset = ExternalDataset(path='./my_dataset', dataset_type='COCO')

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

  # Specify timeout such that optimization is performed for 4 hours
  timeout = 14400

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
  ```

* **Hyperparameter tuning example with the [DetrLearner](detr.md) with a custom study**

  This example shows how to tune a selection of the hyperparameters of the *DetrLearner* and
  how to specify an objective function.

  ```python
  from opendr.utils.hyperparameter_tuner import HyperparameterTuner
  from opendr.perception.object_detection_2d import DetrLearner
  from opendr.engine.datasets import ExternalDataset

  import optuna

  # Create a coco dataset, containing training and evaluation data
  dataset = ExternalDataset(path='./my_dataset', dataset_type='COCO')

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

  # Specify timeout such that optimization is performed for 4 hours
  timeout = 14400

  # Create custom Study
  sampler = optuna.samplers.CmaEsSampler()
  study = optuna.create_study(study_name='detr_cma', sampler=sampler)

  # Initialize the tuner
  tuner = HyperparameterTuner(DetrLearner, study=study)

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
  ```

#### Performance Evaluation

In this section, we will present the performance evaluation of this tool.
This tool is not evaluated quantitatively, since hyperparameter tuning is very problem-specific.
Also, the tool provides an interface with the existing Optuna framework, therefore evaluation of the performance of the hyperparameter tuning tool will be nothing more than evaluating the performance of Optuna.
Quantitative results for Optuna on the Street View House Numbers (SVHN) dataset can be found in [[1]](#optuna-paper).
Rather than providing quantitative results, we will here present an evaluation of the tool in terms of support, features and compatibility.
Below, the supported learner base classes and supported hyperparameter types are presented.
Here it is shown that the hyperparameter tuning tool supports all learners that are present in the OpenDR toolkit.
Also, the hyperparameter types that are supported by Optuna are supported by the tool.
More information on these types can be found [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html).

| Supported Types                                         |
|---------------------------------------------------------|
| All OpenDR Learners (Learner, LearnerRL, LearnerActive) |
| Categorical Hyperparameters                             |
| Discrete Hyperparameters                                |
| Floating Point Hyperparameters                          |
| Integer Hyperparameters                                 |
| Loguniform/Uniform Continous Hyperparameters            |

Below, the sampling algorithms that are available in the tool are shown.
These include both single and multi-objective algorithms.

| Available Sampling Algorithms                                                                               |
|-------------------------------------------------------------------------------------------------------------|
| Grid Sampling                                                                                               |
| Independent Sampling                                                                                        |
| Tree-structured Parzen Estimator (TPE) Sampling                                                             |
| Covariance Matrix Adaptation - Evolution Strategy (CMA-ES) Sampling                                         |
| Partially Fixed Sampling                                                                                    |
| Nondominated Sorting Genetic Algorithm II (NSGA-II) Sampling                                                |
| Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive Optimization (MTSPE) Sampling |

The platform compatibility evaluation is also reported below:

| Platform                                     | Test results |
|----------------------------------------------|:------------:|
| x86 - Ubuntu 20.04 (bare installation - CPU) |     Pass     |
| x86 - Ubuntu 20.04 (bare installation - GPU) |     Pass     |
| x86 - Ubuntu 20.04 (pip installation)        |     Pass     |
| x86 - Ubuntu 20.04 (CPU docker)              |     Pass     |
| x86 - Ubuntu 20.04 (GPU docker)              |     Pass     |
| NVIDIA Jetson TX2                            |     Pass     |

#### References
<a name="optuna-paper" href="https://dl.acm.org/doi/10.1145/3292500.3330701">[1]</a>
Optuna: A Next-generation Hyperparameter Optimization Framework.,
[arXiv](https://arxiv.org/abs/1907.10902).  

# Copyright 2020-2023 OpenDR European Project
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

import os
import argparse
from optuna.trial import TrialState
from opendr.utils.hyperparameter_tuner import HyperparameterTuner
from opendr.perception.object_detection_2d import DetrLearner
from opendr.engine.datasets import ExternalDataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--timeout", help="Stop study after the given number of second(s).", type=float, default=60)

    args = parser.parse_args()

    # Create a coco dataset, containing training and evaluation data
    learner = DetrLearner(device=args.device)
    learner.download(mode='test_data')
    dataset = ExternalDataset(path=os.path.join('temp', 'nano_coco'), dataset_type='COCO')

    # Specify the arguments that are required for the init method
    init_arguments = {'device': args.device}

    # Specify the arguments that are required for the fit method
    fit_arguments = {
        'dataset': dataset,
        'annotations_folder': '',
        'train_annotations_file': 'instances.json',
        'train_images_folder': 'image',
        'silent': True,
    }

    # Specify the arguments that are required for the eval method
    eval_arguments = {
        'dataset': dataset,
        'images_folder': 'image',
        'annotations_folder': '',
        'annotations_file': 'instances.json',
        'verbose': False,
    }

    # Specify duration of the optimization procedure in seconds
    timeout = args.timeout

    # Initialize the tuner
    tuner = HyperparameterTuner(DetrLearner)

    # Optimize
    best_parameters = tuner.optimize(
        init_arguments=init_arguments,
        fit_arguments=fit_arguments,
        eval_arguments=eval_arguments,
        timeout=timeout,
        show_progress_bar=True,
    )

    # Print results
    # Copied from https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
    study = tuner.study
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

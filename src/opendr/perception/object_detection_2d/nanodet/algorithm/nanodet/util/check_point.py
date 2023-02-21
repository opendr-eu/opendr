# Modifications Copyright 2021 - present, OpenDR European Project
#
# Copyright 2021 RangiLyu.
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
from opendr.perception.object_detection_2d.nanodet.algorithm.nanodet.util.rank_filter import rank_filter
from typing import Any, Dict
import torch


def load_model_weight(model, checkpoint, verbose=None):
    state_dict = checkpoint["state_dict"].copy()
    for k in checkpoint["state_dict"]:
        # convert average model weights
        if k.startswith("avg_model."):
            v = state_dict.pop(k)
            state_dict[k[4:]] = v
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in state_dict.items()}

    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                if verbose:
                    print(
                        "Skip loading parameter {}, required shape{}, "
                        "loaded shape{}.".format(
                            k, model_state_dict[k].shape, state_dict[k].shape
                        )
                    )
                state_dict[k] = model_state_dict[k]
        else:
            if verbose:
                print("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            if verbose:
                print("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
    return model


@rank_filter
def save_model(model, path, epoch, iter, optimizer=None):
    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )
    data = {"epoch": epoch, "state_dict": model_state_dict, "iter": iter}
    if optimizer is not None:
        data["optimizer"] = optimizer.state_dict()

    torch.save(data, path)


def save_model_state(path, model, weight_averager=None, verbose=None):
    if verbose:
        print("Saving model to {}".format(path))
    state_dict = (
        weight_averager.state_dict()
        if weight_averager
        else model.state_dict()
    )
    torch.save({"state_dict": state_dict}, path)


def convert_avg_params(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Converts average state dict to the format that can be loaded to a model.
    Args:
        checkpoint: model.
    Returns:
        Converted average state dict.
    """
    state_dict = checkpoint["state_dict"]
    avg_weights = {}
    for k, v in state_dict.items():
        if "avg_model" in k:
            avg_weights[k[10:]] = v
    return avg_weights

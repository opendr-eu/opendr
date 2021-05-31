# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""A function to build localization and classification losses from config."""

from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.pytorch.core import (
    losses,
)
from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.protos import (
    losses_pb2,
)


def build(loss_config, device):
    """Build losses based on the config.

  Builds classification, localization losses and optionally a hard example miner
  based on the config.

  Args:
    loss_config: A losses_pb2.Loss object.

  Returns:
    classification_loss: Classification loss object.
    localization_loss: Localization loss object.
    classification_weight: Classification loss weight.
    localization_weight: Localization loss weight.
    hard_example_miner: Hard example miner object.

  Raises:
    ValueError: If hard_example_miner is used with sigmoid_focal_loss.
  """
    classification_loss = _build_classification_loss(loss_config.classification_loss)
    localization_loss = _build_localization_loss(loss_config.localization_loss, device)
    classification_weight = loss_config.classification_weight
    localization_weight = loss_config.localization_weight
    hard_example_miner = None
    if loss_config.HasField("hard_example_miner"):
        raise ValueError("Pytorch don't support HardExampleMiner")
    return (
        classification_loss,
        localization_loss,
        classification_weight,
        localization_weight,
        hard_example_miner,
    )


def build_faster_rcnn_classification_loss(loss_config):
    """Builds a classification loss for Faster RCNN based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    if not isinstance(loss_config, losses_pb2.ClassificationLoss):
        raise ValueError("loss_config not of type losses_pb2.ClassificationLoss.")

    loss_type = loss_config.WhichOneof("classification_loss")

    if loss_type == "weighted_sigmoid":
        return losses.WeightedSigmoidClassificationLoss()
    if loss_type == "weighted_softmax":
        config = loss_config.weighted_softmax
        return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.logit_scale)

    # By default, Faster RCNN second stage classifier uses Softmax loss
    # with anchor-wise outputs.
    config = loss_config.weighted_softmax
    return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.logit_scale)


def _build_localization_loss(loss_config, device):
    """Builds a localization loss based on the loss config.

  Args:
    loss_config: A losses_pb2.LocalizationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    if not isinstance(loss_config, losses_pb2.LocalizationLoss):
        raise ValueError("loss_config not of type losses_pb2.LocalizationLoss.")

    loss_type = loss_config.WhichOneof("localization_loss")

    if loss_type == "weighted_l2":
        config = loss_config.weighted_l2
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedL2LocalizationLoss(code_weight)

    if loss_type == "weighted_smooth_l1":
        config = loss_config.weighted_smooth_l1
        if len(config.code_weight) == 0:
            code_weight = None
        else:
            code_weight = config.code_weight
        return losses.WeightedSmoothL1LocalizationLoss(config.sigma, code_weight, device)

    raise ValueError("Empty loss config.")


def _build_classification_loss(loss_config):
    """Builds a classification loss based on the loss config.

  Args:
    loss_config: A losses_pb2.ClassificationLoss object.

  Returns:
    Loss based on the config.

  Raises:
    ValueError: On invalid loss_config.
  """
    if not isinstance(loss_config, losses_pb2.ClassificationLoss):
        raise ValueError("loss_config not of type losses_pb2.ClassificationLoss.")

    loss_type = loss_config.WhichOneof("classification_loss")

    if loss_type == "weighted_sigmoid":
        return losses.WeightedSigmoidClassificationLoss()

    if loss_type == "weighted_sigmoid_focal":
        config = loss_config.weighted_sigmoid_focal
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SigmoidFocalClassificationLoss(gamma=config.gamma, alpha=alpha)
    if loss_type == "weighted_softmax_focal":
        config = loss_config.weighted_softmax_focal
        if config.alpha > 0:
            alpha = config.alpha
        else:
            alpha = None
        return losses.SoftmaxFocalClassificationLoss(gamma=config.gamma, alpha=alpha)

    if loss_type == "weighted_softmax":
        config = loss_config.weighted_softmax
        return losses.WeightedSoftmaxClassificationLoss(logit_scale=config.logit_scale)

    if loss_type == "bootstrapped_sigmoid":
        config = loss_config.bootstrapped_sigmoid
        return losses.BootstrappedSigmoidClassificationLoss(
            alpha=config.alpha,
            bootstrap_type=("hard" if config.hard_bootstrap else "soft"),
        )

    raise ValueError("Empty loss config.")

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

from opendr.engine.learners import Learner
from opendr.engine.datasets import DatasetIterator
from opendr.engine.target import BoundingBox3DList
from opendr.perception.object_tracking_3d.ab3dmot.algorithm.ab3dmot import AB3DMOT
from opendr.perception.object_tracking_3d.ab3dmot.algorithm.evaluate import evaluate as evaluate_kitti_tracking
from opendr.perception.object_tracking_3d.ab3dmot.logger import Logger


class ObjectTracking3DAb3dmotLearner(Learner):
    def __init__(
        self,
        device="cpu",
        max_staleness=2,
        min_updates=3,
        state_dimensions=10,  # x, y, z, rotation_y, l, w, h, speed_x, speed_z, angular_speed
        measurement_dimensions=7,  # x, y, z, rotation_y, l, w, h
        state_transition_matrix=None,
        measurement_function_matrix=None,
        covariance_matrix=None,
        process_uncertainty_matrix=None,
        iou_threshold=0.01,
    ):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(ObjectTracking3DAb3dmotLearner, self).__init__(
            device=device,
        )

        self.max_staleness = max_staleness
        self.min_updates = min_updates
        self.state_dimensions = state_dimensions
        self.measurement_dimensions = measurement_dimensions
        self.state_transition_matrix = state_transition_matrix
        self.measurement_function_matrix = measurement_function_matrix
        self.covariance_matrix = covariance_matrix
        self.process_uncertainty_matrix = process_uncertainty_matrix
        self.iou_threshold = iou_threshold

        self.__create_model()

    def save(self, path):
        raise NotImplementedError("The AB3DMOT Learner does not support saving")

    def load(
        self,
        path,
    ):
        raise NotImplementedError("The AB3DMOT Learner does not support loading")

    def reset(self):
        self.model.reset()

    def fit(
        self,
        dataset,
        val_dataset=None,
        logging_path=None,
        silent=False,
        verbose=False,
    ):
        raise NotImplementedError("The AB3DMOT Learner does not support training")

    def eval(
        self,
        dataset,
        logging_path=None,
        silent=False,
        verbose=False,
        count=None
    ):

        logger = Logger(silent, verbose, logging_path)

        if not isinstance(dataset, DatasetIterator):
            raise ValueError("dataset should be a DatasetIterator")

        if count is None:
            count = len(dataset)

        predictions = []
        ground_truths = []

        for i in range(count):
            self.reset()
            input, ground_truth = dataset[i]
            predictions.append(self.infer(input))
            ground_truths.append(ground_truth)

            logger.log(Logger.LOG_WHEN_NORMAL, "Computing tracklets [" + str(i + 1) + "/" + str(count) + "]", end='\r')

        result = evaluate_kitti_tracking(predictions, ground_truths, log=logger.log)

        logger.close()

        return result

    def infer(self, bounding_boxes_3d_list):

        if self.model is None:
            raise ValueError("No model created")

        is_single_input = True

        if isinstance(bounding_boxes_3d_list, BoundingBox3DList):
            bounding_boxes_3d_list = [bounding_boxes_3d_list]
        elif isinstance(bounding_boxes_3d_list, list):
            is_single_input = False
        else:
            return ValueError(
                "bounding_boxes_3d_list should be a BoundingBox3DList or a list of BoundingBox3DList"
            )

        results = []

        for box_list in bounding_boxes_3d_list:
            result = self.model.update(box_list)
            results.append(result)

        if is_single_input:
            results = results[0]

        return results

    def optimize(self):
        raise Exception("The AB3DMOT Learner does not support optimization")

    def __create_model(self):

        self.model = AB3DMOT(
            frame=0,
            max_staleness=self.max_staleness,
            min_updates=self.min_updates,
            state_dimensions=self.state_dimensions,
            measurement_dimensions=self.measurement_dimensions,
            state_transition_matrix=self.state_transition_matrix,
            measurement_function_matrix=self.measurement_function_matrix,
            covariance_matrix=self.covariance_matrix,
            process_uncertainty_matrix=self.process_uncertainty_matrix,
            iou_threshold=self.iou_threshold,
        )

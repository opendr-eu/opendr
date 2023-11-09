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


class RealTimeEvaluator:
    def __init__(
        self,
        data_fps,
        require_predictive_inference=False,
        wait_for_next_frame=False,
        cap_model_fps=None,
    ) -> None:
        self.data_fps = data_fps
        self.require_predictive_inference = require_predictive_inference  # label at t is compared to prediction before t
        self.wait_for_next_frame = wait_for_next_frame  # when last data frame is already old, we can wait for the next one
        self.cap_model_fps = cap_model_fps
        self.time_since_last_frame = 0
        self.labels = []
        self.predictions = []

    @property
    def data_delta_time(self):
        return 1 / self.data_fps

    def model_delta_time(self, delta_time):

        if delta_time == 0:
            return 0

        return delta_time if self.cap_model_fps is None else max(delta_time, (1 / self.cap_model_fps))

    def on_data(
        self, label, frame
    ):  # labels are expected to come each self.data_delta_time seconds
        self.labels.append((label, frame))

        if (not self.require_predictive_inference) and len(self.labels) <= 1:
            # Shift labels to allow comparision of prediction at frame t with the label at frame t,
            # if the FPS is higher than the data FPS.
            self.labels.append((label, frame))

        for prediction in self.predictions:
            prediction[1] = max(0, prediction[1] - self.data_delta_time)

        while len(self.predictions) > 1 and self.predictions[1][1] <= 0:
            self.predictions.pop(0)

        label_to_compare, frame_to_compare = self.labels.pop(0)
        prediction_to_compare = self.predictions[0][0]

        return label_to_compare, prediction_to_compare, frame_to_compare, self.predictions[0][2]

    def on_prediction(self, prediction, model_delta_time, frame):
        self.predictions.append(
            [prediction, self.time_since_last_frame + self.model_delta_time(model_delta_time), frame]
        )
        self.time_since_last_frame = max(
            0,
            self.time_since_last_frame +
            self.model_delta_time(model_delta_time)
        )

    def can_frame_be_processed(self):  # expected to be called each frame

        self.time_since_last_frame = max(
            0,
            self.time_since_last_frame -
            self.data_delta_time,
        )

        return self.time_since_last_frame <= (self.data_delta_time / 2 if self.wait_for_next_frame else self.data_delta_time)

    def init(self, label, prediction):
        self.time_since_last_frame = 0
        self.labels = []
        self.predictions = []
        self.on_prediction(prediction, 0, -1)
        self.on_data(label, -1)

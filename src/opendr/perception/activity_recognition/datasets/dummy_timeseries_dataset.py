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

import torch
from opendr.engine.datasets import DatasetIterator


class DummyTimeseriesDataset(DatasetIterator, torch.utils.data.Dataset):
    """
    Dumme dataset for time-series forecasting

    Input data: Sinusoidal data of different wavelength
    Targets: Last sum of cosines quantized into four buckets
        (positive and falling, positive and rising, negative and falling, negative and rising)
    """

    def __init__(self, num_sines=4, num_datapoints=64, base_offset=0, sequence_len=64):
        DatasetIterator.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        time_steps = torch.stack(
            [
                torch.stack(
                    [
                        torch.arange(
                            0 + offset + base_offset,
                            i + offset + base_offset,
                            i / sequence_len,
                        )
                        for i in range(1, num_sines + 1)
                    ],
                    dim=1,
                )
                for offset in range(num_datapoints)
            ]
        ).permute(0, 2, 1)
        self._input_data = torch.sin(time_steps)
        assert self._input_data.shape == (num_datapoints, num_sines, sequence_len)

        cosines = torch.cos(time_steps[:, :-2]).sum(dim=-1)
        positive = cosines[:, -1] > 0
        upwards = cosines[:, -1] > cosines[:, -2]
        self._output_data = positive + 2 * upwards

    def __getitem__(self, idx):
        return self._input_data[idx], self._output_data[idx]

    def __len__(self):
        return len(self._output_data)

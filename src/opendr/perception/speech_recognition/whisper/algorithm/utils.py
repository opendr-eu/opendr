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


from typing import List


def matching_percentage(hypothesis: List[str], reference: List[str]) -> float:
    """
    Compute the accuracy of string predicted by the model and the ground truth.
    Used in keyword matching.

    Args:
        hypothesis (List[str]): A list of predicted strings.
        reference (List[str]): A list of ground truth strings.

    Returns:
        float: The accuracy of the predicted strings.

    Raises:
        AssertionError: If the model is not loaded.
    """

    if len(hypothesis) != len(reference):
        raise ValueError("Both lists must have the same length.")

    matching_count = sum(h == r for h, r in zip(hypothesis, reference))
    total_count = len(hypothesis)

    return matching_count / total_count

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

import os
from perception.object_tracking_2d.deep_sort.object_tracking_2d_deep_sort_learner import (
    ObjectTracking2DDeepSortLearner,
)
from perception.object_tracking_2d.datasets.market1501_dataset import (
    Market1501Dataset,
    Market1501DatasetIterator,
)
from perception.object_tracking_2d.datasets.mot_dataset import (
    RawMotWithDetectionsDatasetIterator,
)


def test():

    dataset = Market1501Dataset(
        "/home/io/OpenDR/opendr_internal/tests/sources" +
        "/tools/perception/object_tracking_2d/deep_sort/deep_sort_temp/nano_market1501"
    )

    learner = ObjectTracking2DDeepSortLearner()
    learner.fit(dataset, 40, verbose=True)

    print()


def test2():

    dataset = Market1501DatasetIterator(
        "/home/io/OpenDR/opendr_internal/tests/sources" +
        "/tools/perception/object_tracking_2d/deep_sort/deep_sort_temp/nano_market1501/bounding_box_train"
    )

    learner = ObjectTracking2DDeepSortLearner()
    result = learner.fit(
        dataset, 40, val_dataset=dataset, verbose=True, val_epochs=1
    )

    print(result)
    print()


def test3():

    train_split_paths = {
        "nano_mot20": os.path.join(
            ".", "perception", "object_tracking_2d",
            "datasets", "splits", "nano_mot20.train"
        )
    }

    dataset = RawMotWithDetectionsDatasetIterator(
        "/home/io/OpenDR/opendr_internal/tests/sources/tools/perception/object_tracking_2d/deep_sort/deep_sort_temp/dataset",
        train_split_paths
    )

    r = dataset[0]

    learner = ObjectTracking2DDeepSortLearner()
    result = learner.infer(r[0])

    print(result)
    print()


def test4():

    train_split_paths = {
        "nano_mot20": os.path.join(
            ".", "perception", "object_tracking_2d",
            "datasets", "splits", "nano_mot20.train"
        )
    }

    dataset = RawMotWithDetectionsDatasetIterator(
        "/home/io/OpenDR/opendr_internal/tests/sources/tools/perception/object_tracking_2d/deep_sort/deep_sort_temp/dataset",
        train_split_paths
    )

    learner = ObjectTracking2DDeepSortLearner()
    result = learner.eval(dataset)

    print(result)
    print()

test4()

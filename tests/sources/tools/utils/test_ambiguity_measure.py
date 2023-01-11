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

import unittest
import numpy as np
from opendr.engine.data import Image
from opendr.utils.ambiguity_measure.ambiguity_measure import AmbiguityMeasure


class TestAmbiguityMeasure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n\n**********************************\nTEST Ambiguity Measure\n" "**********************************")
        cls.am = AmbiguityMeasure()

    def test_get_ambiguity_measure(self):
        heatmap = 10 * np.random.random((128, 128))
        ambiguous, locs, maxima, probs = self.am.get_ambiguity_measure(heatmap)
        self.assertTrue(type(ambiguous) in [bool, np.bool_])
        self.assertTrue(type(locs) in [list, np.ndarray])
        self.assertTrue(type(maxima) in [list, np.ndarray])
        self.assertTrue(type(probs) in [list, np.ndarray])

    def test_plot_ambiguity_measure(self):
        img = 255 * np.random.random((128, 128, 3))
        img = np.asarray(img, dtype="uint8")
        heatmap = 10 * np.random.random((128, 128))
        ambiguous, locs, maxima, probs = self.am.get_ambiguity_measure(heatmap)
        self.am.plot_ambiguity_measure(heatmap, locs, probs, img)

        img = Image(img)
        self.am.plot_ambiguity_measure(heatmap, locs, probs, img)

    def test_threshold(self):
        threshold = self.am.threshold
        new_threshold = threshold * 0.2
        self.am.threshold = new_threshold
        self.assertTrue(self.am.threshold == new_threshold)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python

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

"""Test that checks that all the source files have the Apache 2 license."""

import unittest
import os
import fnmatch
import datetime

from io import open

APACHE2_LICENSE_C = """/*
 * Copyright 2020-20XX OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */""".replace('20XX', str(datetime.datetime.now().year))

APACHE2_LICENSE_CPP = """// Copyright 2020-20XX OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.""".replace('20XX', str(datetime.datetime.now().year))

APACHE2_LICENSE_PYTHON = """# Copyright 2020-20XX OpenDR European Project
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
# limitations under the License.""".replace('20XX', str(datetime.datetime.now().year))

PYTHON_OPTIONAL_HEADERS = [
    '#!/usr/bin/env python2',
    '#!/usr/bin/env python3',
    '#!/usr/bin/env python',
]


class TestLicense(unittest.TestCase):
    """Unit test for checking that all the source files have the Apache 2 license."""

    def setUp(self):
        """Get all the source files which require a license check."""
        directories = [
            'src',
            'projects',
            'include'
        ]

        skippedDirectoryPaths = [
            'src/opendr/perception/pose_estimation/lightweight_open_pose/algorithm',
            'src/opendr/perception/object_detection_3d/voxel_object_detection_3d/second_detector',
            'src/opendr/perception/face_recognition/algorithm',
            'src/opendr/perception/activity_recognition/x3d/algorithm',
            'src/opendr/perception/activity_recognition/cox3d/algorithm',
            'src/opendr/perception/object_tracking_2d/fair_mot/algorithm',
            'src/opendr/perception/object_tracking_2d/deep_sort/algorithm',
            'src/opendr/perception/object_tracking_2d/siamrpn/data_utils',
            'src/opendr/perception/compressive_learning/multilinear_compressive_learning/algorithm/backbones',
            'src/opendr/perception/heart_anomaly_detection/attention_neural_bag_of_feature/algorithm',
            'src/opendr/simulation/human_model_generation/utilities/PIFu',
            'src/opendr/perception/multimodal_human_centric/rgbd_hand_gesture_learner/algorithm/architectures',
            'src/opendr/perception/skeleton_based_action_recognition/algorithm',
            'projects/python/simulation/synthetic_multi_view_facial_image_generation/algorithm',
            'projects/opendr_ws/devel',
            'src/opendr/perception/semantic_segmentation/bisenet/algorithm',
            'src/opendr/perception/object_detection_2d/retinaface/algorithm',
            'src/opendr/perception/object_detection_2d/gem/algorithm',
            'src/opendr/perception/object_detection_2d/detr/algorithm',
            'src/opendr/perception/object_detection_2d/nanodet/algorithm',
            'src/opendr/perception/panoptic_segmentation/efficient_ps/algorithm/EfficientPS',
            'src/opendr/perception/panoptic_segmentation/efficient_lps/algorithm/EfficientLPS',
            'src/opendr/perception/facial_expression_recognition/landmark_based_facial_expression_recognition/algorithm',
            'src/opendr/perception/facial_expression_recognition/image_based_facial_emotion_estimation/algorithm',
            'projects/python/perception/facial_expression_recognition/image_based_facial_emotion_estimation',
            'projects/opendr_ws_2/src/opendr_perception/test',
            'projects/opendr_ws_2/src/opendr_ros2_bridge/test',
            'projects/opendr_ws_2/src/vision_opencv',
            'projects/opendr_ws_2/install',
            'projects/opendr_ws_2/src/data_generation/test',
            'projects/opendr_ws_2/src/opendr_planning/test',
            'projects/opendr_ws_2/src/opendr_bridge/test',
            'projects/opendr_ws_2/src/opendr_interface/test',
            'projects/opendr_ws_2/src/opendr_data_generation/test',
            'projects/opendr_ws_2/src/opendr_simulation/test',
        ]

        skippedFilePaths = [
            'src/opendr/perception/activity_recognition/datasets/utils/decoder.py',
            'projects/python/perception/pose_estimation/lightweight_open_pose/jetbot/utils/pid.py',
            'src/opendr/perception/compressive_learning/multilinear_compressive_learning/algorithm/trainers.py',
            'src/opendr/perception/object_detection_2d/retinaface/Makefile',
            'src/opendr/perception/multimodal_human_centric/audiovisual_emotion_learner/algorithm/efficientface_modulator.py',
            'src/opendr/perception/multimodal_human_centric/audiovisual_emotion_learner/algorithm/efficientface_utils.py',
            'src/opendr/perception/multimodal_human_centric/audiovisual_emotion_learner/algorithm/spatial_transforms.py',
            'src/opendr/perception/multimodal_human_centric/audiovisual_emotion_learner/algorithm/transformer_timm.py',
            'src/opendr/perception/multimodal_human_centric/audiovisual_emotion_learner/algorithm/utils.py',
            'projects/opendr_ws_2/src/opendr_perception/setup.py',
            'projects/opendr_ws_2/src/opendr_planning/setup.py',
            'projects/opendr_ws_2/src/opendr_bridge/setup.py',
            'projects/opendr_ws_2/src/data_generation/setup.py',
            'projects/opendr_ws_2/src/opendr_simulation/setup.py',
        ]

        skippedDirectories = [
            'build'
        ]

        extensions = ['*.c', '*.cpp', '*.h', '*.hpp', '*.py', '*.java', 'Makefile']

        self.sources = []
        for directory in directories:
            for rootPath, dirNames, fileNames in os.walk(os.environ['OPENDR_HOME'] + os.sep + directory.replace('/', os.sep)):
                shouldContinue = False
                relativeRootPath = rootPath.replace(os.environ['OPENDR_HOME'] + os.sep, '')
                for path in skippedDirectoryPaths:
                    if rootPath.startswith(os.environ['OPENDR_HOME'] + os.sep + path.replace('/', os.sep)):
                        shouldContinue = True
                        break
                currentDirectories = rootPath.replace(os.environ['OPENDR_HOME'], '').split(os.sep)
                for directory in skippedDirectories:
                    if directory in currentDirectories:
                        shouldContinue = True
                        break
                if shouldContinue:
                    continue
                for extension in extensions:
                    for fileName in fnmatch.filter(fileNames, extension):
                        if fileName == '__init__.py':
                            continue
                        if os.path.join(relativeRootPath, fileName).replace(os.sep, '/') in skippedFilePaths:
                            continue
                        file = os.path.join(rootPath, fileName)
                        self.sources.append(file)

    def test_sources_have_license(self):
        """Test that sources have the license."""
        for source in self.sources:
            with open(source, 'r', encoding='utf-8') as content_file:
                content = content_file.read()
                if source.endswith('.c') or source.endswith('.h'):
                    self.assertTrue(
                        APACHE2_LICENSE_C in content,
                        msg='Source file "%s" doesn\'t contain the correct Apache 2.0 License:\n%s' %
                            (source, APACHE2_LICENSE_C)
                    )
                elif source.endswith('.cpp') or source.endswith('.hpp') or source.endswith('.java'):
                    self.assertTrue(
                        APACHE2_LICENSE_CPP in content,
                        msg='Source file "%s" doesn\'t contain the correct Apache 2.0 License:\n%s' %
                            (source, APACHE2_LICENSE_CPP)
                    )
                elif source.endswith('.py') or source.endswith('Makefile'):
                    self.assertTrue(
                        APACHE2_LICENSE_PYTHON in content,
                        msg='Source file "%s" doesn\'t contain the correct Apache 2.0 License:\n%s' %
                            (source, APACHE2_LICENSE_PYTHON)
                    )
                else:
                    self.assertTrue(
                        False,
                        msg='Unsupported file extension "%s".' % source
                    )


if __name__ == '__main__':
    unittest.main()

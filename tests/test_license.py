#!/usr/bin/env python

# Copyright 2020-2021 Cyberbotics Ltd.
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

from io import open

APACHE2_LICENSE_C = """* Licensed under the Apache License, Version 2.0 (the "License");
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
 */"""

APACHE2_LICENSE_CPP = """// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License."""

APACHE2_LICENSE_PYTHON = """
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
# limitations under the License."""

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
            'src/opendr/perception/compressive_learning/multilinear_compressive_learning/algorithm/backbones',
            'src/opendr/perception/object_detection_2d/retinaface/algorithm',
        ]

        skippedFilePaths = [
            'src/opendr/perception/activity_recognition/datasets/utils/decoder.py',
            'projects/perception/lightweight_open_pose/jetbot/utils/pid.py',
            'src/opendr/perception/compressive_learning/multilinear_compressive_learning/algorithm/trainers.py',
            'src/opendr/perception/object_detection_2d/retinaface/Makefile',
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

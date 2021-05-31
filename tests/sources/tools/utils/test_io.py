# Copyright 2020-2021 OpenDR Project
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
import shutil
import opendr.utils.io as io
from pathlib import Path
from logging import getLogger
import json

logger = getLogger(__name__)


class TestIO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(
            "./tests/sources/tools/utils/temp"
        )
        cls.temp_dir.mkdir(exist_ok=True, parents=True)

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(str(cls.temp_dir))
        except OSError as e:
            logger.error(f"Caught error while cleaning up {e.filename}: {e.strerror}")

    def test_bump_version(self):

        def write_data(path: Path):
            with path.open("w", encoding="utf-8") as f:
                json.dump({"dummy": 42}, f)

        p1 = self.temp_dir / "dummy_file.json"
        p2 = self.temp_dir / "dummy_file_1.json"
        p3 = self.temp_dir / "dummy_file_2.json"

        # Ensure files don't exist beforehand
        try:
            p1.unlink()
            p2.unlink()
            p3.unlink()
        except FileNotFoundError:
            pass

        # Non existing file
        assert io.bump_version(p1) == p1
        write_data(p1)
        assert p1.exists()

        # File exists (with no '_x' prepended)
        assert not p2.exists()
        assert io.bump_version(p1) == p2
        write_data(p2)
        assert p2.exists()

        # File exists (with '_x' prepended)
        assert not p3.exists()
        assert io.bump_version(p1) == p3
        assert io.bump_version(p2) == p3


if __name__ == "__main__":
    unittest.main()

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

from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve
import tarfile
import os
import time
import sys

OPENDR_HOME = os.environ["OPENDR_HOME"]


def download_data(raw_data_only):
    def reporthook(count, block_size, total_size):
        nonlocal start_time
        nonlocal last_print

        if count == 0:
            start_time = time.time()
            last_print = start_time
            return

        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        if time.time() - last_print >= 1:
            last_print = time.time()
            print(
                "\r%d MB, %d KB/s, %d seconds passed" %
                (progress_size / (1024 * 1024), speed, duration),
                end=''
            )

    human_data_url = OPENDR_SERVER_URL + "simulation/SMPLD_body_models/human_models.tar.gz"
    downloaded_human_data_path = os.path.join(OPENDR_HOME, 'projects/python/simulation/SMPL+D_human_models/human_models.tar.gz')
    print("Downloading data from", human_data_url, "to", downloaded_human_data_path)
    start_time = 0
    last_print = 0
    urlretrieve(human_data_url, downloaded_human_data_path, reporthook=reporthook)

    def is_within_directory(directory, target):
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
        prefix = os.path.commonprefix([abs_directory, abs_target])

        return prefix == abs_directory

    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")

        tar.extractall(path, members, numeric_owner=numeric_owner)

    with tarfile.open(downloaded_human_data_path) as tar:
        safe_extract(tar, path=os.path.join(OPENDR_HOME, 'projects/python/simulation/SMPL+D_human_models'))
    tar.close()
    os.remove(downloaded_human_data_path)

    if raw_data_only:
        return

    model_url = OPENDR_SERVER_URL + "simulation/SMPLD_body_models/model.tar.gz"
    downloaded_model_path = os.path.join(OPENDR_HOME, 'projects/python/simulation/SMPL+D_human_models/model.tar.gz')
    print("Downloading data from", model_url, "to", downloaded_model_path)
    start_time = 0
    last_print = 0
    urlretrieve(model_url, downloaded_model_path, reporthook=reporthook)
    with tarfile.open(downloaded_model_path) as tar:
        safe_extract(tar, path=os.path.join(OPENDR_HOME, 'projects/python/simulation/SMPL+D_human_models'))
    tar.close()
    os.remove(downloaded_model_path)


if __name__ == "__main__":
    raw_data = False
    if len(sys.argv) > 1 and sys.argv[1] == 'raw':
        raw_data = True
    download_data(raw_data)

# Copyright 2020-2022 OpenDR European Project
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

import json
import argparse
import glob
from urllib.request import urlretrieve
import os
import warnings


def search_on_path(filenames):
    for p in os.environ.get('PATH', '').split(os.pathsep):
        for filename in filenames:
            full = os.path.join(p, filename)
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


def get_cuda_path():
    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'))
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(os.path.join(os.path.dirname(nvcc_path), '..'))
        _cuda_path = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path = '/usr/local/cuda'
    else:
        _cuda_path = None

    return _cuda_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_path", help="Path to installed cuda", type=str, default=None)
    parser.add_argument("--opendr_device", help="Target device for installation",
                        type=str, choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--torch_version", help="Specifies LibTorch version to be installed", type=str, default="1.9.0")
    args = parser.parse_args()

    COMPATIBILITY_VERSIONS = {
        "1.13.1": "0.14.1",
        "1.13.0": "0.14.0",
        "1.12.0": "0.13.0",
        "1.11.0": "0.12.0",
        "1.10.2": "0.11.3",
        "1.10.1": "0.11.2",
        "1.10.0": "0.11.1",
        "1.9.1": "0.10.1",
        "1.9.0": "0.10.0",
    }

    warnings.simplefilter("error")

    TORCH_VERSION = args.torch_version
    VISION_VERSION = COMPATIBILITY_VERSIONS[TORCH_VERSION]

    CUDA_VERSION = None
    DEVICE = None
    # Find Device
    if args.opendr_device == "gpu":
        try:
            if args.cuda_path is None:
                CUDA_PATH = get_cuda_path()
            else:
                CUDA_PATH = args.cuda_path
            version_file_type = glob.glob(f"{CUDA_PATH}/version*")
            if version_file_type[0].endswith('.txt'):
                version_file = open(f"{CUDA_PATH}/version.txt", mode='r')
                version_line = version_file.readlines()
                version_line = version_line[0].replace(".", "")
                CUDA_VERSION = version_line[13:16]
                version_file.close()
            elif version_file_type[0].endswith('.json'):
                version_file = open(f"{CUDA_PATH}/version.json", mode='r')
                version_dict = json.load(version_file)
                CUDA_VERSION = version_dict["cuda"]["version"]
                CUDA_VERSION = CUDA_VERSION.replace(".", "")
                CUDA_VERSION = CUDA_VERSION[:3]
                version_file.close()
            else:
                warnings.warn("\033[93m No CUDA version file found.")
            DEVICE = f"cu{CUDA_VERSION}"
        except:
            warnings.warn("\033[93m No CUDA installation found.\n"
                          "Please install CUDA or specify CUDA path with export CUDA_PATH=/path/to/your/cuda.")
            exit()
    else:
        DEVICE = "cpu"

    # Download Libtorch
    try:
        file_url_libtorch = f"https://download.pytorch.org/libtorch/{DEVICE}/" \
                   f"libtorch-cxx11-abi-shared-with-deps-{TORCH_VERSION}%2B{DEVICE}.zip"

        DOWNLOAD_DIRECTORY = "libtorch.zip"

        urlretrieve(file_url_libtorch, DOWNLOAD_DIRECTORY)

    except:
        warnings.warn("\033[93m No LibTorch found for your specific device and torch version.\n"
                      "Please choose another version of torch or install a different version of CUDA.\n"
                      "Please refer to https://download.pytorch.org/whl/torch_stable.html")
        exit()
    # Download Vision
    try:
        file_url_vision = f"https://github.com/pytorch/vision/archive/refs/tags/" \
                          f"v{VISION_VERSION}.tar.gz"
        DOWNLOAD_DIRECTORY = "vision.tar.gz"
        urlretrieve(file_url_vision, DOWNLOAD_DIRECTORY)
    except:
        warnings.warn("\033[93m No torchvision found for your specific torch version.\n"
                      "Please refer to https://github.com/pytorch/vision for more information.")
        exit()


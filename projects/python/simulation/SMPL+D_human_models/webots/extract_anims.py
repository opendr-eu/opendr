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

import os
import numpy as np
import sys


def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError('Path to database is not provided.')
    dirName = sys.argv[1]
    listOfFiles = getListOfFiles(dirName)
    for elem in listOfFiles:
        if elem.split('.')[-1] == 'npz':
            with np.load(elem) as data:
                if not os.path.isdir(elem.split('.')[0]):
                    os.mkdir(elem.split('.')[0])
                np.save(os.path.join(elem.split('.')[0], 'betas.npy'), data['betas'])
                np.save(os.path.join(elem.split('.')[0], 'gender.npy'), data['gender'])
                np.save(os.path.join(elem.split('.')[0], 'poses.npy'), data['poses'])
                np.save(os.path.join(elem.split('.')[0], 'trans.npy'), data['trans'])
                if 'mocap_framerate' in data:
                    np.save(os.path.join(elem.split('.')[0], 'mocap_framerate.npy'), data['mocap_framerate'])
                np.save(os.path.join(elem.split('.')[0], 'dmpls.npy'), data['dmpls'])
            os.remove(elem)

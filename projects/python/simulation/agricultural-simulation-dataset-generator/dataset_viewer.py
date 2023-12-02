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

import cv2

cv2.namedWindow('UGV-image', cv2.WINDOW_NORMAL)
cv2.namedWindow('UGV-segmentation', cv2.WINDOW_NORMAL)

cv2.namedWindow('UAV-image', cv2.WINDOW_NORMAL)
cv2.namedWindow('UAV-segmentation', cv2.WINDOW_NORMAL)

path_ugv = 'dataset_location/UGV/'
path_uav = 'dataset_location/UAV/'

i = 1  # second
index = 1
index_uav = 0
counter = 1
for _ in range(6000):
    try:
        index_one = str(index)
        if ((index) % 3 == 0):
            print(index)
            image_filename_ugv = path_ugv + 'front_bottom_camera/' + str(i) + '_' + index_one.zfill(6) + '.jpg'
            print(image_filename_ugv)
            image_ugv = cv2.imread(image_filename_ugv)
            cv2.imshow('UGV-image', image_ugv)
            # cv2.waitKey(1)

        if ((index) % 3 == 0):
            print(index)
            image_filename_ugv = path_ugv + 'annotations/front_bottom_camera/' + str(i) + '_' + index_one.zfill(
                6) + '_segmented.jpg'
            print(image_filename_ugv)
            image_ugv = cv2.imread(image_filename_ugv)
            cv2.imshow('UGV-segmentation', image_ugv)
            # cv2.waitKey(1)

        index_one_uav = str(index_uav)
        if ((index_uav) % 3 == 0):
            print(index)
            image_filename_uav = path_uav + 'camera/' + str(i) + '_' + index_one_uav.zfill(6) + '.jpg'
            print(image_filename_uav)
            image_uav = cv2.imread(image_filename_uav)
            cv2.imshow('UAV-image', image_uav)
            cv2.waitKey(1)

        if ((index_uav) % 3 == 0):
            print(index)
            image_filename_uav = path_uav + 'annotations/camera/' + str(i) + '_' + index_one_uav.zfill(
                6) + '_segmented.jpg'
            print(image_filename_uav)
            image_uav = cv2.imread(image_filename_uav)
            cv2.imshow('UAV-segmentation', image_uav)
            cv2.waitKey(1)

        index += 1
        index_uav += 1

        counter += 1

        if ((counter) % 100 == 0):
            print(index)
            print('increase i')
            counter = 1
            index_uav = 1
            i += 1

    except (AttributeError, KeyError):
        print('err')

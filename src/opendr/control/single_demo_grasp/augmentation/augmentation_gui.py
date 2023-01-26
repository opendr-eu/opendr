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

import sys
import os
import numpy as np
from math import random
import cv2
from augmentation_utils import \
    annotate, Augment_train_straight_box_n_kps, BoundingBoxesOnImage, BoundingBox, KeypointsOnImage, Keypoint

'''
generate Data randomly from imgaug library

when test == 1, only one example of augmentation is made for verification only
when test == 0, the augmentation is done based on the value of train_scale and val_scale
'''

test = 0
object_name = "pendulum"
img_path = "image.png"
imge = cv2.imread(img_path, 1)
imge = cv2.resize(imge, (640, 480), interpolation=cv2.INTER_AREA)
x = np.array(imge)
new_img = np.expand_dims(x, axis=0)

if test == 1:

    train_scale = 1
    val_scale = 1

else:

    train_scale = 1500
    val_scale = 200


directory_save = os.path.join('datasets', object_name)

try:
    os.makedirs(os.path.join(directory_save, 'annotations'))
except FileExistsError:
    pass
try:
    os.makedirs(os.path.join(directory_save, 'train'))
except FileExistsError:
    pass
try:
    os.makedirs(os.path.join(directory_save, 'val'))
except FileExistsError:
    pass


# annotate object
bbx_points, grasp_line = annotate(imge)
img_aug_train, boxes_train, kps_train = Augment_train_straight_box_n_kps(object_name, new_img, train_scale,
                                                                         bbx_points, grasp_line, "train", test)

_, boxes_val, kps_val = Augment_train_straight_box_n_kps(object_name, new_img, val_scale,
                                                         bbx_points, grasp_line, "val", test)


np.save(os.path.join(directory_save, 'annotations', 'boxes_train.npy'), boxes_train)
np.save(os.path.join(directory_save, 'annotations', 'boxes_val.npy'), boxes_val)
np.save(os.path.join(directory_save, 'annotations', 'kps_train.npy'), kps_train)
np.save(os.path.join(directory_save, 'annotations', 'kps_val.npy'), kps_val)

i = random.randint(0, train_scale - 1)
test_box = boxes_train[i]
test_kps = kps_train[i]

bbs = BoundingBoxesOnImage([BoundingBox(x1=int(test_box[0]), y1=int(test_box[1]), x2=int(test_box[2]), y2=int(test_box[3]))],
                           shape=imge.shape)
image_after = bbs.draw_on_image(img_aug_train[i][0], size=2, color=[0, 255, 255])

a = test_kps[0]
b = test_kps[1]

kypt = KeypointsOnImage(list(Keypoint(x=x_in, y=y_in) for (x_in, y_in) in zip(a, b)),
                        shape=imge.shape)

image_after = kypt.draw_on_image(image_after, size=7, color=[0, 0, 255])
cv2.destroyAllWindows()
cv2.imshow("test", image_after)
cv2.waitKey(500)
print("press enter to close the augmentation program")
input()
cv2.destroyAllWindows()
sys.exit()

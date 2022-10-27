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

import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
import random
from PIL import Image

# User input GUI utilities *


def click_event(event, x, y, flags, params):
    kps = params[1]
    bbx = params[0]
    img = params[2]
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(bbx) == 0:

            bbx.append((x, y))
            center_coordinates = (x, y)
            radius = 10
            color = (0, 255, 0)
            thickness = 2

            cv2.circle(img, center_coordinates, radius, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 255, 255), 2)
            cv2.imshow('image', img)
            print("please select the opposite corner of the bounding box on the image")

        elif len(bbx) == 1:

            bbx.append((x, y))
            center_coordinates = (x, y)
            radius = 10
            color = (0, 255, 0)
            thickness = 2

            cv2.circle(img, center_coordinates, radius, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 255, 255), 2)

            start_point = bbx[0]
            end_point = bbx[1]
            color = (0, 255, 0)
            thickness = 2
            image_plt = cv2.rectangle(img, start_point, end_point, color, thickness)
            cv2.imshow('image', image_plt)
            print("please select the first keypoint on the image")

    if event == cv2.EVENT_RBUTTONDOWN:

        if len(kps) == 0 and len(bbx) == 2:

            kps.append((x, y))
            center_coordinates = (x, y)
            radius = 10
            color = (0, 0, 255)
            thickness = 2

            cv2.circle(img, center_coordinates, radius, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 255, 255), 2)
            cv2.imshow('image', img)
            print("please select the second keypoint on the image")

        elif len(kps) == 1 and len(bbx) == 2:

            kps.append((x, y))
            center_coordinates = (x, y)
            radius = 10
            color = (0, 0, 255)
            thickness = 2

            cv2.circle(img, center_coordinates, radius, color, thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 1, (255, 255, 255), 2)

            start_point = kps[0]
            end_point = kps[1]
            color = (0, 0, 255)
            thickness = 2
            cv2.line(img, start_point, end_point, color, thickness)
            cv2.imshow('image', img)
            print("Done, press any key on keyboard to continue augmentation")


def annotate(img):

    bbx = []
    kps = []

    cv2.namedWindow("image", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow('image', 640, 480)
    cv2.imshow('image', img)
    print("please select the first corner of the bounding box on the image")
    params = [bbx, kps, img]
    cv2.setMouseCallback('image', click_event, params)
    cv2.waitKey(0)

    bbx_out = [min(bbx[0][0], bbx[1][0]), min(bbx[0][1], bbx[1][1]),
               max(bbx[0][0], bbx[1][0]), max(bbx[0][1], bbx[1][1])]
    kps_out = [kps[0][0], kps[0][1], kps[1][0], kps[1][1]]

    return bbx_out, kps_out


# Augmentation  utilities *

def Augment_train_straight_box_n_kps(object_name, images, scale, bbx_in, grasp_line, group, test):

    print("augmentation started...")
    img_aug = []
    boxes = []
    kps_aug = []

    num_new_points = 10
    kps_x_in = np.linspace(grasp_line[0], grasp_line[2], num_new_points + 2)
    kps_y_in = np.linspace(grasp_line[1], grasp_line[3], num_new_points + 2)

    for batch_idx in range(scale):
        bbs = BoundingBoxesOnImage([
            BoundingBox(x1=bbx_in[0], y1=bbx_in[1], x2=bbx_in[2], y2=bbx_in[3])], shape=images[0].shape)

        kps = KeypointsOnImage(list(Keypoint(x=x_in, y=y_in) for (x_in, y_in) in
                               zip(kps_x_in, kps_y_in)), shape=images[0].shape)

        if test == 1:
            rt = 0
        else:
            rt = random.randint(-180, 180)

        seq = iaa.Sequential([iaa.Affine(rotate=rt), iaa.Multiply((0.4, 1.5), per_channel=0.99),
                             iaa.Affine(translate_px={"x": (-30, 30), "y": (-30, 30)}), iaa.Affine(scale=(0.75, 1.3))])
        batch = seq(images=images, bounding_boxes=bbs, keypoints=kps, return_batch=True)
        image_aug = batch.images_aug
        bbs_aug = batch.bounding_boxes_aug
        kypt_aug = batch.keypoints_aug
        key_list_x = []
        key_list_y = []

        for i in range(len(kypt_aug.keypoints)):
            key_list_x.append(kypt_aug.keypoints[i].x)
            key_list_y.append(kypt_aug.keypoints[i].y)

        key_list_x = np.array(key_list_x)
        key_list_y = np.array(key_list_y)

        kps_aug.append([key_list_x, key_list_y])

        img_aug.append(image_aug)
        after = bbs_aug.bounding_boxes[0]
        boxes.append((after.x1, after.y1, after.x2, after.y2))
        img = Image.fromarray(image_aug[0], 'RGB')
        outdir = "datasets/" + object_name + "/" + group + "/" + str(batch_idx) + ".jpg"
        img.save(outdir)

    return img_aug, boxes, kps_aug

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
#
import os
import cv2
import numpy as np
import json
from gluoncv.data.tracking_data import AnchorTarget
from gluoncv.data.transforms.track import SiamRPNaugmentation
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import center2corner
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import Center
from opendr.engine.datasets import DatasetIterator


class OTBTrainDataset(DatasetIterator):
    def __init__(self, root, json_path, train_exemplar_size=127, train_search_size=255,
                 train_output_size=17, anchor_stride=8, anchor_ratios=(0.33, 0.5, 1, 2, 3),
                 train_thr_high=0.6, train_thr_low=0.3, train_pos_num=16, train_neg_num=16,
                 train_total_num=64, template_shift=4, template_scale=0.05, template_blur=0,
                 template_flip=0, template_color=1.0, search_shift=64, search_scale=0.18,
                 search_blur=0, search_flip=0, search_color=1.0):
        super(OTBTrainDataset, self).__init__()
        # dataset specific reading functions
        self.root = root
        self.json_path = json_path
        with open(self.json_path, 'r') as f:
            self.annotations = json.load(f)
        self.video_names = list(self.annotations.keys())
        self.n_videos = len(self.video_names)
        print(f'Found {self.n_videos} videos, reading data...')

        # SiamRPN functionality
        self.train_exemplar_size = train_exemplar_size
        self.train_search_size = train_search_size
        self.train_output_size = train_output_size

        # create anchor target
        self.anchor_target = AnchorTarget(anchor_stride=anchor_stride,
                                          anchor_ratios=anchor_ratios,
                                          train_search_size=self.train_search_size,
                                          train_output_size=self.train_output_size,
                                          train_thr_high=train_thr_high,
                                          train_thr_low=train_thr_low,
                                          train_pos_num=train_pos_num,
                                          train_neg_num=train_neg_num,
                                          train_total_num=train_total_num)

        # data augmentation
        self.template_aug = SiamRPNaugmentation(template_shift,
                                                template_scale,
                                                template_blur,
                                                template_flip,
                                                template_color)
        self.search_aug = SiamRPNaugmentation(search_shift,
                                              search_scale,
                                              search_blur,
                                              search_flip,
                                              search_color)

    def __len__(self):
        return self.n_videos

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.train_exemplar_size
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def _get_crop(self, image, box):
        image_mean = np.mean(image, axis=(0, 1))
        image = crop_like_siamfc(image, box, exemplar_size=self.train_exemplar_size,
                                 search_size=self.train_search_size, padding=image_mean)
        return image

    def __getitem__(self, item):
        video_name = self.video_names[item]

        # choose two random frames from video
        available_frames = self.annotations[video_name]["img_names"]
        indices = np.random.choice(len(available_frames), 2)
        frame1, frame2 = available_frames[indices[0]], available_frames[indices[1]]

        # read frames and transform to target, search
        template_image = cv2.imread(os.path.join(self.root, frame1))
        search_image = cv2.imread(os.path.join(self.root, frame2))
        # crop around boxes
        template_box = self.annotations[video_name]["gt_rect"][indices[0]]
        search_box = self.annotations[video_name]["gt_rect"][indices[1]]
        template_image = self._get_crop(template_image, template_box)
        search_image = self._get_crop(search_image, search_box)

        # get corresponding bounding boxes
        # we only need the dimensions of the boxes (w, h) as we have already cropped the images around the targets
        template_box = self._get_bbox(template_image, template_box[2:])
        search_box = self._get_bbox(search_image, search_box[2:])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        self.train_exemplar_size)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       self.train_search_size)

        # get labels
        cls, delta, delta_weight, _ = self.anchor_target(bbox, self.train_output_size, False)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)

        return template, search, cls, delta, delta_weight, np.array(bbox)


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_siamfc(image, bbox, exemplar_size=127, context_amount=0.5, search_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[0] + bbox[2] / 2.), (bbox[1] + bbox[3] / 2.)]
    target_size = [bbox[2], bbox[3]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return x

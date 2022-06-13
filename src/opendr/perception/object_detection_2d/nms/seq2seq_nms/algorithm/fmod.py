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

import torch
import torchvision
import numpy as np
import cv2
import random
from opendr.engine.data import Image


class FMoD:
    def __init__(self, roi_pooling_dim=None, pyramid_depth=3, map_type="SIFT", map_bin=False,
                 resize_dim=None, device='cpu'):
        if roi_pooling_dim is None:
            roi_pooling_dim = 160
        self.roi_pooling_dim = [roi_pooling_dim, roi_pooling_dim]
        self.pyramid_depth = pyramid_depth
        self.boxes_p = []
        self.rp_size = []
        for p in range(self.pyramid_depth):
            s = 1 / pow(2, p)
            for i in np.arange(0, 1.0, s):
                for j in np.arange(0, 1.0, s):
                    self.boxes_p.append([0, int(i * self.roi_pooling_dim[0]), int(j * self.roi_pooling_dim[1]),
                                         int((i + s) * self.roi_pooling_dim[0]),
                                         int((j + s) * self.roi_pooling_dim[1])])
            self.rp_size.append([int(self.roi_pooling_dim[0] * s), int(self.roi_pooling_dim[1] * s)])
        self.device = device
        self.boxes_p = torch.tensor(self.boxes_p).float()
        if "cuda" in self.device:
            self.boxes_p = self.boxes_p.to(self.device)
        self.resc = 1.0
        self.map = None
        self.resize_dim = resize_dim
        self.map_type = map_type
        self.map_bin = map_bin
        self.mean = None
        self.std = None

    def set_mean_std(self, mean_values=None, std_values=None):
        self.mean = torch.tensor(mean_values).float()
        self.std = torch.tensor(std_values).float()
        if "cuda" in self.device:
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def extract_maps(self, img=None, augm=False):
        if img is None:
            raise Exception('Image is not provided to FMoD...')

        if not isinstance(img, Image):
            img = Image(img)
        img = img.convert(format='channels_last', channel_order='bgr')

        if self.resize_dim is not None:
            max_dim = max(img.shape[0], img.shape[1])
            if max_dim > self.resize_dim:
                self.resc = float(self.resize_dim) / max_dim
                img = cv2.resize(img, (int(img.shape[1] * self.resc), int(img.shape[0] * self.resc)))
        if augm:
            img = augm_brightness(img, 0.75, 1.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.map_type == "EDGEMAP":
            dst_img = np.copy(img)
            dst_img = cv2.GaussianBlur(dst_img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
            gradX = cv2.Scharr(dst_img, ddepth=cv2.CV_16S, dx=1, dy=0, scale=1, delta=0,
                               borderType=cv2.BORDER_DEFAULT)
            gradY = cv2.Scharr(dst_img, ddepth=cv2.CV_16S, dx=0, dy=1, scale=1, delta=0,
                               borderType=cv2.BORDER_DEFAULT)
            absGradX = cv2.convertScaleAbs(gradX)
            absGradY = cv2.convertScaleAbs(gradY)
            absGradXCV32 = absGradX.astype("float32")
            absGradYCV32 = absGradY.astype("float32")
            self.map = cv2.magnitude(absGradXCV32 / 255.0, absGradYCV32 / 255.0)
            self.map = self.map * 255
            if self.map_bin:
                self.map = cv2.threshold(self.map, 240, 255, cv2.THRESH_BINARY)[1]
        else:
            kps = None
            if self.map_type == "FAST":
                fast = cv2.FastFeatureDetector_create()
                kps = fast.detect(img, None)
            elif self.map_type == "AKAZE":
                akaze = cv2.AKAZE_create()
                kps, desc = akaze.detectAndCompute(img, None)
            elif self.map_type == "BRISK":
                brisk = cv2.BRISK_create()
                kps = brisk.detect(img, None)
            elif self.map_type == "ORB":
                orb = cv2.ORB_create()
                kps = orb.detect(img, None)
            else:
                raise Exception("Map type not supported...")
            self.map = np.zeros(img.shape, dtype=np.uint8)
            coords_x = []
            coords_y = []
            resps = []
            for kp in kps:
                coords_x.append(int(kp.pt[0]))
                coords_y.append(int(kp.pt[1]))
                resps.append(255 * kp.response)
            if not self.map_bin:
                self.map[coords_y, coords_x] = resps
            else:
                self.map[coords_y, coords_x] = 255
        self.map = torch.from_numpy(self.map).float()
        if "cuda" in self.device:
            self.map = self.map.to(self.device)

    def extract_FMoD_feats(self, boxes):
        num_rois = boxes.shape[0]
        map_gpu = self.map / 255.0
        map_gpu = map_gpu.unsqueeze(0).unsqueeze(0)
        descs = []
        pooled_regions = torchvision.ops.roi_align(map_gpu, [self.resc * boxes],
                                                   output_size=self.rp_size[0], spatial_scale=1.0,
                                                   aligned=True)
        pooled_regions = pooled_regions.unsqueeze(1)
        descs.append(self.get_descriptor(pooled_regions))
        for i in range(0, self.pyramid_depth - 1):
            pooled_regions_pyr = pooled_regions.contiguous().view(num_rois, pooled_regions.shape[-2],
                                                                  pooled_regions.shape[-1])
            pooled_regions_pyr = pooled_regions_pyr.unsqueeze(0)
            pooled_regions_pyr = torchvision.ops.roi_align(pooled_regions_pyr, self.boxes_p[(pow(4 + 1, i)):(
                    (pow(4 + 1, i)) + pow(4, (i + 1))), :], output_size=self.rp_size[i + 1], aligned=True)
            pooled_regions_pyr = pooled_regions_pyr.permute(1, 0, 2, 3)
            pooled_regions_pyr = pooled_regions_pyr.contiguous().view(num_rois, 1, pooled_regions_pyr.shape[-3],
                                                                      pooled_regions_pyr.shape[-2],
                                                                      pooled_regions_pyr.shape[-1])
            descs.append(self.get_descriptor(pooled_regions_pyr))

        descs = torch.cat(descs, dim=1)
        if self.mean is not None and self.std is not None:
            descs = (descs - self.mean) / self.std
            descs = torch.clamp(descs, -50, 50)
        return descs

    def release_maps(self):
        self.map = None

    def get_descriptor(self, patches):
        dt = []
        # row data
        dt.append(patches.mean(dim=3))
        # collumn data
        dt.append(patches.mean(dim=4))
        # block data
        dt.append(torch.flatten(patches, start_dim=3))

        means = []
        stds = []
        diffs = []
        zscores = []
        skews = []
        kurtoses = []
        powers = []
        for i in range(len(dt)):
            if i == 2:
                means.append(dt[i].mean(dim=3))
            else:
                means.append(dt[i][:, :, :, 0:-1:5].mean(dim=3))
            stds.append(dt[i].std(dim=3))
            diffs.append((dt[i] - means[i].unsqueeze(-1).expand(dt[i].size())))
            zscores.append(diffs[i] / stds[i].unsqueeze(-1).expand(dt[i].size()))
            zscores[i] = torch.where(stds[i].unsqueeze(-1).expand(zscores[i].shape) > 0, zscores[i],
                                     torch.zeros_like(zscores[i]))
            skews.append(torch.mean(torch.pow(zscores[i], 3.0), -1))
            kurtoses.append(torch.mean(torch.pow(zscores[i], 4.0), -1) - 3.0)
            powers.append((dt[i] * dt[i]).mean(-1))
        descs = []
        for i in range(len(dt)):
            descs.append(torch.cat((means[i], stds[i], skews[i], kurtoses[i], powers[i]), 2))
        desc = torch.cat((descs[0], descs[1], descs[2]), 2)
        desc = desc.contiguous().view(desc.shape[0], desc.shape[1] * desc.shape[2])
        return desc


def augm_brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

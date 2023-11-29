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

# General imports

import torchvision.transforms
import os
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm

from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.data import Image
from opendr.engine.target import Pose
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR lightweight_open_pose imports
from opendr.perception.pose_estimation.lightweight_open_pose.lightweight_open_pose_learner import \
    LightweightOpenPoseLearner
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.load_state import \
    load_state
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.keypoints import \
    extract_keypoints, group_keypoints
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.val import \
    convert_to_coco_format, run_coco_eval, normalize, pad_width


class HighResolutionPoseEstimationLearner(LightweightOpenPoseLearner):

    def __init__(self, device='cuda', backbone='mobilenet',
                 temp_path='temp', mobilenet_use_stride=True, mobilenetv2_width=1.0, shufflenet_groups=3,
                 num_refinement_stages=2, batches_per_iter=1, base_height=256,
                 first_pass_height=360, second_pass_height=540, method='adaptive', percentage_around_crop=0.3,
                 heatmap_threshold=0.1, experiment_name='default', num_workers=8, weights_only=True,
                 output_name='detections.json', multiscale=False, scales=None, visualize=False,
                 img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256), pad_value=(0, 0, 0),
                 half_precision=False):

        super(HighResolutionPoseEstimationLearner, self).__init__(device=device, backbone=backbone, temp_path=temp_path,
                                                                  mobilenet_use_stride=mobilenet_use_stride,
                                                                  mobilenetv2_width=mobilenetv2_width,
                                                                  shufflenet_groups=shufflenet_groups,
                                                                  num_refinement_stages=num_refinement_stages,
                                                                  batches_per_iter=batches_per_iter,
                                                                  base_height=base_height,
                                                                  experiment_name=experiment_name,
                                                                  num_workers=num_workers, weights_only=weights_only,
                                                                  output_name=output_name, multiscale=multiscale,
                                                                  scales=scales, visualize=visualize, img_mean=img_mean,
                                                                  img_scale=img_scale, pad_value=pad_value,
                                                                  half_precision=half_precision)

        self.first_pass_height = first_pass_height
        self.second_pass_height = second_pass_height
        self.method = method
        self.perc = percentage_around_crop
        self.threshold = heatmap_threshold
        self.prev_heatmap = np.array([])
        self.counter = 0
        if self.method == 'primary':
            self.xmin = None
            self.ymin = None
            self.xmax = None
            self.ymax = None

        elif self.method == 'adaptive':
            self.xmin = None
            self.ymin = None
            self.xmax = None
            self.ymax = None

            self.x1min = None
            self.x1max = None
            self.y1min = None
            self.y1max = None

            self.x2min = None
            self.x2max = None
            self.y2min = None
            self.y2max = None

    def __first_pass(self, img):
        """
        This method is generating a rough heatmap of the input image in order to specify the approximate location
        of humans in the picture.

        :param img: input image for heatmap generation
        :type img: numpy.ndarray

        :return: returns the Part Affinity Fields (PAFs) of the humans inside the image
        :rtype: numpy.ndarray
        """

        if 'cuda' in self.device:
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            tensor_img = tensor_img.cuda()
            if self.half:
                tensor_img = tensor_img.half()
        else:
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cpu()

        stages_output = self.model(tensor_img)
        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        return pafs

    def __second_pass(self, img, net_input_height_size, max_width, stride, upsample_ratio,
                      pad_value=(0, 0, 0),
                      img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
        """
        This method detects the keypoints and estimates the pose of humans using the cropped image from the
        previous step (__first_pass_).

        :param img: input image for heatmap generation
        :type img: numpy.ndarray
        :param net_input_height_size: the height that the input image will be resized  for inference
        :type net_input_height_size: int
        :param max_width: this parameter is the maximum width that the resized image should have. It is introduced to
            avoid cropping images with abnormal ratios e.g (30, 800)
        :type max_width: int
        :param upsample_ratio: Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing,
            defaults to 4
        :type upsample_ratio: int, optional

         :returns: the heatmap of human figures, the part affinity filed (pafs), the scale of the resized image compared
            to the initial and the pad around the image
         :rtype: heatmap, pafs -> numpy.ndarray
                 scale -> float
                 pad = -> list
        """

        height, width, _ = img.shape
        scale = net_input_height_size / height
        img_ratio = width / height
        if img_ratio > 6:
            scale = max_width / width

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        if 'cuda' in self.device:
            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            tensor_img = tensor_img.cuda()
            if self.half:
                tensor_img = tensor_img.half()
        else:
            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().cpu()

        stages_output = self.model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = heatmaps.astype(np.float32)
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = pafs.astype(np.float32)
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def __pooling(self, img, kernel):  # Pooling on input image for dimension reduction
        """This method applies a pooling filter on an input image in order to resize it in a fixed shape

         :param img: input image for resizing
         :rtype img: engine.data.Image class object
         :param kernel: the kernel size of the pooling filter
         :type kernel: int
         """
        pool_img = torchvision.transforms.ToTensor()(img)
        if 'cuda' in self.device:
            pool_img = pool_img.cuda()
            if self.half:
                pool_img = pool_img.half()
        pool_img = pool_img.unsqueeze(0)
        pool_img = torch.nn.functional.avg_pool2d(pool_img, kernel)
        pool_img = pool_img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return pool_img

    @staticmethod
    def __crop_heatmap(heatmap):
        """
        This method takes the generated heatmap and crops it around the desirable ROI using its nonzero values.

        :param heatmap: the heatmap that generated from __first_pass function
        :type heatmap: numpy.array

        :returns An array that contains the boundaries of the cropped image
        :rtype: np.array
        """
        detection = False

        if heatmap.nonzero()[0].size > 10 and heatmap.nonzero()[0].size > 10:
            detection = True
            xmin = min(heatmap.nonzero()[1])
            ymin = min(heatmap.nonzero()[0])
            xmax = max(heatmap.nonzero()[1])
            ymax = max(heatmap.nonzero()[0])
        else:
            xmin, ymin, xmax, ymax = 0, 0, 0, 0

        heatmap_dims = (int(xmin), int(xmax), int(ymin), int(ymax))
        return heatmap_dims, detection

    @staticmethod
    def __check_for_split(cropped_heatmap):
        """
        This function checks weather or not the cropped heatmap needs further proccessing for extra cropping.
        More specifically, returns a boolean for the decision for further crop, the decision depends on the distance between the
        target subjects.

        :param cropped_heatmap: the cropped area from the original heatmap
        :type cropped_heatmap: np.array

        :returns: A boolean that describes weather is needed to proceed on further cropping
        :rtype: bool
        """
        sum_rows = cropped_heatmap.sum(axis=1)
        sum_col = cropped_heatmap.sum(axis=0)

        heatmap_activation_area = len(sum_col.nonzero()[0]) * len(sum_rows.nonzero()[0])
        crop_total_area = cropped_heatmap.shape[0] * cropped_heatmap.shape[1]  # heatmap total area

        if crop_total_area != 0:
            crop_rule1 = (heatmap_activation_area / crop_total_area * 100 < 80) and (
                    heatmap_activation_area / crop_total_area * 100 > 5)
            if crop_rule1:
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def __split_process(cropped_heatmap):
        """
        This function uses the cropped heatmap that crated from __crop_heatmap function and splits it in parts.

        :param cropped_heatmap: the cropped area from the original heatmap
        :type cropped_heatmap: np.array

        :returns: Returns a list with the new dimensions of the split parts
        :rtype: list
        """
        max_diff_c, max_diff_r = 0, 0
        y_crop_l = cropped_heatmap.shape[1]
        y_crop_r = 0
        sum_col = cropped_heatmap.sum(axis=0)
        x_crop_u = 0
        sum_rows = cropped_heatmap.sum(axis=1)

        for ind in range(len(sum_col.nonzero()[0]) - 1):
            diff_c = abs(sum_col.nonzero()[0][ind + 1] - sum_col.nonzero()[0][ind])
            if (diff_c > max_diff_c) and (diff_c > 5):  # nonzero columns have at least 5px difference in heatmap
                max_diff_c = diff_c
                y_crop_l = round(sum_col.nonzero()[0][ind])
                y_crop_r = round(sum_col.nonzero()[0][ind + 1])

        for ind in range(len(sum_rows.nonzero()[0]) - 1):
            diff_r = abs(sum_rows.nonzero()[0][ind + 1] - sum_rows.nonzero()[0][ind])
            if (diff_r > max_diff_r) and (diff_r > 5):  # nonzero rows have at least 5px difference in heatmap
                max_diff_r = diff_r
                x_crop_u = round(sum_rows.nonzero()[0][ind])

        if max_diff_c >= max_diff_r and max_diff_c > 0:     # vertical cut
            y1_i = 0
            y1_f = cropped_heatmap.shape[0]
            x1_i = 0
            x1_f = int(y_crop_l)

            y2_i = 0
            y2_f = cropped_heatmap.shape[0]
            x2_i = int(y_crop_r)
            x2_f = cropped_heatmap.shape[1]

            crop1 = cropped_heatmap[y1_i:y1_f, x1_i:x1_f]
            crop2 = cropped_heatmap[y2_i:y2_f, x2_i:x2_f]

        elif max_diff_r > max_diff_c and max_diff_r > 0:    # horizontal cut
            y1_i = 0
            y1_f = int(x_crop_u)
            x1_i = 0
            x1_f = cropped_heatmap.shape[1]

            y2_i = int(x_crop_u + 3)
            y2_f = cropped_heatmap.shape[0]
            x2_i = 0
            x2_f = cropped_heatmap.shape[1]

            crop1 = cropped_heatmap[y1_i:y1_f, x1_i:x1_f]
            crop2 = cropped_heatmap[y2_i:y2_f, x2_i:x2_f]

        else:
            return [[cropped_heatmap, 0, cropped_heatmap.shape[1], 0, cropped_heatmap.shape[0]]]

        crops = [[crop1, x1_i, x1_f, y1_i, y1_f], [crop2, x2_i, x2_f, y2_i, y2_f]]
        return crops

    @staticmethod
    def __crop_enclosing_bbox(crop):
        """
        This function creates the bounding box for each split part

        :param crop: A split part from the original heatmap
        :type crop: np.array

        :returns:  the dimensions (xmin, xmax, ymin, ymax) for enclosing bounding box
        :rtype: int
        """
        if crop.nonzero()[0].size > 0 and crop.nonzero()[1].size > 0:
            xmin = min(np.unique(crop.nonzero()[1]))
            ymin = min(np.unique(crop.nonzero()[0]))
            xmax = max(np.unique(crop.nonzero()[1]))
            ymax = max(np.unique(crop.nonzero()[0]))
        else:
            xmin, xmax, ymin, ymax = 0, 0, 0, 0
        return xmin, xmax, ymin, ymax

    @staticmethod
    def __crop_image_func(xmin, xmax, ymin, ymax, pool_img, original_image, heatmap, perc):
        """
        This function crops the region of interst(ROI) from the original image to use it in next steps

        :param xmin, ymin: top left corner dimensions of the split part in the original heatmap
        :type xmin,ymin: int
        :param xmax, ymax: bottom right dimensions of the split part in the original heatmap
        :type xmin,ymin: int
        :param pool_img: the resized pooled input image
        :type pool_img: np.array
        :param original_image: the original input image
        :type original_image: np.array
        :param heatmap: the heatmap generated from __first_pass function
        :type heatmap: np.array
        :param perc: percentage of the image that is needed for adding extra pad
        :type perc: float

        :returns: Returns the cropped image part from the original image and the dimensions of the cropped part in the
        original image coordinate system
        :rtype :numpy.array, int, int, int, int
        """
        upscale_factor_x = pool_img.shape[0] / heatmap.shape[0]
        upscale_factor_y = pool_img.shape[1] / heatmap.shape[1]
        xmin = upscale_factor_x * xmin
        xmax = upscale_factor_x * xmax
        ymin = upscale_factor_y * ymin
        ymax = upscale_factor_y * ymax

        upscale_to_init_img = original_image.shape[0] / pool_img.shape[0]
        xmin = upscale_to_init_img * xmin
        xmax = upscale_to_init_img * xmax
        ymin = upscale_to_init_img * ymin
        ymax = upscale_to_init_img * ymax

        extra_pad_x = int(perc * (xmax - xmin))  # Adding an extra pad around cropped image
        extra_pad_y = int(perc * (ymax - ymin))

        if xmin - extra_pad_x > 0:
            xmin = xmin - extra_pad_x
        else:
            xmin = xmin
        if xmax + extra_pad_x < original_image.shape[1]:
            xmax = xmax + extra_pad_x
        else:
            xmax = xmax

        if ymin - extra_pad_y > 0:
            ymin = ymin - extra_pad_y
        else:
            ymin = ymin
        if ymax + extra_pad_y < original_image.shape[0]:
            ymax = ymax + extra_pad_y
        else:
            ymax = ymax

        if (xmax - xmin) > 40 and (ymax - ymin) > 40:
            crop_img = original_image[int(ymin):int(ymax), int(xmin):int(xmax)]
        else:
            crop_img = original_image

        return crop_img, int(xmin), int(xmax), int(ymin), int(ymax)

    def fit(self, dataset, val_dataset=None, logging_path='', logging_flush_secs=30,
            silent=False, verbose=True, epochs=None, use_val_subset=True, val_subset_size=250,
            images_folder_name="train2017", annotations_filename="person_keypoints_train2017.json",
            val_images_folder_name="val2017", val_annotations_filename="person_keypoints_val2017.json"):
        """This method is not used in this implementation."""

        raise NotImplementedError

    def optimize(self, do_constant_folding=False):
        """This method is not used in this implementation."""

        raise NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def save(self, path, verbose=False):
        """This method is not used in this implementation."""
        return NotImplementedError

    def eval(self, dataset, silent=False, verbose=True, use_subset=True, subset_size=250, upsample_ratio=4,
             images_folder_name="val2017", annotations_filename="person_keypoints_val2017.json"):
        """
                This method is used to evaluate a trained model on an evaluation dataset.

                :param dataset: object that holds the evaluation dataset.
                :type dataset: ExternalDataset class object or DatasetIterator class object
                :param silent: if set to True, disables all printing of evaluation progress reports and other
                    information to STDOUT, defaults to 'False'
                :type silent: bool, optional
                :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
                :type verbose: bool, optional
                :param use_subset: If set to True, a subset of the validation dataset is created and used in
                    evaluation, defaults to 'True'
                :type use_subset: bool, optional
                :param subset_size: Controls the size of the validation subset, defaults to '250'
                :type subset_size: int, optional
                param upsample_ratio: Defines the amount of upsampling to be performed on the heatmaps and PAFs
                    when resizing,defaults to 4
                :type upsample_ratio: int, optional
                :param images_folder_name: Folder name that contains the dataset images. This folder should be contained
                in the dataset path provided. Note that this is a folder name, not a path, defaults to 'val2017'
                :type images_folder_name: str, optional
                :param annotations_filename: Filename of the annotations json file. This file should be contained in the
                    dataset path provided, defaults to 'person_keypoints_val2017.json'
                :type annotations_filename: str, optional

                :returns: returns stats regarding evaluation
                :rtype: dict
                """

        data = super(HighResolutionPoseEstimationLearner,  # NOQA
                     self)._LightweightOpenPoseLearner__prepare_val_dataset(dataset, use_subset=use_subset,
                                                                            subset_name="val_subset.json",
                                                                            subset_size=subset_size,
                                                                            images_folder_default_name=images_folder_name,
                                                                            annotations_filename=annotations_filename,
                                                                            verbose=verbose and not silent)
        # Model initialization if needed
        if self.model is None and self.checkpoint_load_iter != 0:
            # No model loaded, initializing new
            self.init_model()
            # User set checkpoint_load_iter, so they want to load a checkpoint
            # Try to find the checkpoint_load_iter checkpoint
            checkpoint_name = "checkpoint_iter_" + str(self.checkpoint_load_iter) + ".pth"
            checkpoints_folder = os.path.join(self.parent_dir, '{}_checkpoints'.format(self.experiment_name))
            full_path = os.path.join(checkpoints_folder, checkpoint_name)
            try:
                checkpoint = torch.load(full_path, map_location=torch.device(self.device))
            except FileNotFoundError as e:
                e.strerror = "File " + checkpoint_name + " not found inside checkpoints_folder, " \
                                                         "provided checkpoint_load_iter (" + \
                             str(self.checkpoint_load_iter) + \
                             ") doesn't correspond to a saved checkpoint.\nNo such file or directory."
                raise e
            if not silent and verbose:
                print("Loading checkpoint:", full_path)

            load_state(self.model, checkpoint)
        elif self.model is None:
            raise AttributeError("self.model is None. Please load a model or set checkpoint_load_iter.")

        self.model = self.model.eval()  # Change model state to evaluation
        self.model.to(self.device)
        if "cuda" in self.device:
            self.model = self.model.to(self.device)
            if self.half:
                self.model.half()

        if self.multiscale:
            self.scales = [0.5, 1.0, 1.5, 2.0]

        coco_result = []
        num_keypoints = Pose.num_kpts

        pbar_eval = None
        if not silent:
            pbar_desc = "Evaluation progress"
            pbar_eval = tqdm(desc=pbar_desc, total=len(data), bar_format="{l_bar}%s{bar}{r_bar}" % '\x1b[38;5;231m')

        for sample in data:
            file_name = sample['file_name']
            img = sample['img']
            h, w, _ = img.shape
            max_width = w
            kernel = int(h / self.first_pass_height)
            if kernel > 0:
                pool_img = self.__pooling(img, kernel)
            else:
                pool_img = img

            avg_pafs = self.__first_pass(pool_img)      # Heatmap Generation
            avg_pafs = avg_pafs.astype(np.float32)

            pafs_map = cv2.blur(avg_pafs, (5, 5))
            pafs_map[pafs_map < self.threshold] = 0

            heatmap = pafs_map.sum(axis=2)
            heatmap = heatmap * 100
            heatmap = heatmap.astype(np.uint8)
            heatmap = cv2.blur(heatmap, (5, 5))

            contours, hierarchy = cv2.findContours(heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            count = []
            coco_keypoints = []

            if len(contours) > 0:
                for x in contours:
                    count.append(x)
                xdim = []
                ydim = []

                for j in range(len(count)):  # Loop for every human (every contour)
                    for i in range(len(count[j])):
                        xdim.append(count[j][i][0][0])
                        ydim.append(count[j][i][0][1])

                h, w, _ = pool_img.shape
                xmin = int(np.floor(min(xdim))) * int((w / heatmap.shape[1])) * kernel
                xmax = int(np.floor(max(xdim))) * int((w / heatmap.shape[1])) * kernel
                ymin = int(np.floor(min(ydim))) * int((h / heatmap.shape[0])) * kernel
                ymax = int(np.floor(max(ydim))) * int((h / heatmap.shape[0])) * kernel

                extra_pad_x = int(self.perc * (xmax - xmin))  # Adding an extra pad around cropped image
                extra_pad_y = int(self.perc * (ymax - ymin))

                if xmin - extra_pad_x > 0:
                    xmin = xmin - extra_pad_x
                if xmax + extra_pad_x < img.shape[1]:
                    xmax = xmax + extra_pad_x
                if ymin - extra_pad_y > 0:
                    ymin = ymin - extra_pad_y
                if ymax + extra_pad_y < img.shape[0]:
                    ymax = ymax + extra_pad_y

                if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                    crop_img = img[ymin:ymax, xmin:xmax]
                else:
                    crop_img = img[0:img.shape[0], 0:img.shape[1]]

                h, w, _ = crop_img.shape

                # ------- Second pass of the image, inference for pose estimation -------
                avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, self.second_pass_height, max_width,
                                                                        self.stride, upsample_ratio)
                total_keypoints_num = 0
                all_keypoints_by_type = []
                for kpt_idx in range(18):
                    total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                             total_keypoints_num)

                pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                for kpt_id in range(all_keypoints.shape[0]):
                    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / upsample_ratio - pad[1]) / scale
                    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / upsample_ratio - pad[0]) / scale

                for i in range(all_keypoints.shape[0]):
                    for j in range(all_keypoints.shape[1]):
                        if j == 0:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                        if j == 1:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

                current_poses = []
                for n in range(len(pose_entries)):
                    if len(pose_entries[n]) == 0:
                        continue
                    pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                    for kpt_id in range(num_keypoints):
                        if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    pose = Pose(pose_keypoints, pose_entries[n][18])
                    current_poses.append(pose)

                coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

                image_id = int(file_name[0:file_name.rfind('.')])

                for idx in range(len(coco_keypoints)):
                    coco_result.append({
                        'image_id': image_id,
                        'category_id': 1,  # person
                        'keypoints': coco_keypoints[idx],
                        'score': scores[idx]
                    })

            if self.visualize:
                for keypoints in coco_keypoints:
                    for idx in range(len(keypoints) // 3):
                        cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                                   3, (255, 0, 255), -1)
                cv2.imshow('keypoints', img)
                key = cv2.waitKey()
                if key == 27:  # esc
                    return
            if not silent:
                pbar_eval.update(1)

        with open(self.output_name, 'w') as f:
            json.dump(coco_result, f, indent=4)
        if len(coco_result) != 0:
            if use_subset:
                result = run_coco_eval(os.path.join(dataset.path, "val_subset.json"),
                                       self.output_name, verbose=not silent)
            else:
                result = run_coco_eval(os.path.join(dataset.path, annotations_filename),
                                       self.output_name, verbose=not silent)
            return {"average_precision": result.stats[0:5], "average_recall": result.stats[5:]}
        else:
            if not silent and verbose:
                print("Evaluation ended with no detections.")
            return {"average_precision": [0.0 for _ in range(5)], "average_recall": [0.0 for _ in range(5)]}

    def eval_adaptive(self, dataset, silent=False, verbose=True, use_subset=True, subset_size=250, upsample_ratio=4,
                      images_folder_name="val2017", annotations_filename="person_keypoints_val2017.json"):
        """
        This method is used to evaluate a trained model on an evaluation dataset.

        :param dataset: object that holds the evaluation dataset.
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param silent: if set to True, disables all printing of evaluation progress reports and other
            information to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :param use_subset: If set to True, a subset of the validation dataset is created and used in
            evaluation, defaults to 'True'
        :type use_subset: bool, optional
        :param subset_size: Controls the size of the validation subset, defaults to '250'
        :type subset_size: int, optional
        param upsample_ratio: Defines the amount of upsampling to be performed on the heatmaps and PAFs
            when resizing,defaults to 4
        :type upsample_ratio: int, optional
        :param images_folder_name: Folder name that contains the dataset images. This folder should be contained
        in the dataset path provided. Note that this is a folder name, not a path, defaults to 'val2017'
        :type images_folder_name: str, optional
        :param annotations_filename: Filename of the annotations json file. This file should be contained in the
            dataset path provided, defaults to 'person_keypoints_val2017.json'
        :type annotations_filename: str, optional

        :returns: returns stats regarding evaluation
        :rtype: dict
        """

        data = super(HighResolutionPoseEstimationLearner,  # NOQA
                     self)._LightweightOpenPoseLearner__prepare_val_dataset(dataset, use_subset=use_subset,
                                                                            subset_name="val_subset.json",
                                                                            subset_size=subset_size,
                                                                            images_folder_default_name=images_folder_name,
                                                                            annotations_filename=annotations_filename,
                                                                            verbose=verbose and not silent)
        # Model initialization if needed
        if self.model is None and self.checkpoint_load_iter != 0:
            # No model loaded, initializing new
            self.init_model()
            # User set checkpoint_load_iter, so they want to load a checkpoint
            # Try to find the checkpoint_load_iter checkpoint
            checkpoint_name = "checkpoint_iter_" + str(self.checkpoint_load_iter) + ".pth"
            checkpoints_folder = os.path.join(self.parent_dir, '{}_checkpoints'.format(self.experiment_name))
            full_path = os.path.join(checkpoints_folder, checkpoint_name)
            try:
                checkpoint = torch.load(full_path, map_location=torch.device(self.device))
            except FileNotFoundError as e:
                e.strerror = "File " + checkpoint_name + " not found inside checkpoints_folder, " \
                                                         "provided checkpoint_load_iter (" + \
                             str(self.checkpoint_load_iter) + \
                             ") doesn't correspond to a saved checkpoint.\nNo such file or directory."
                raise e
            if not silent and verbose:
                print("Loading checkpoint:", full_path)

            load_state(self.model, checkpoint)
        elif self.model is None:
            raise AttributeError("self.model is None. Please load a model or set checkpoint_load_iter.")

        self.model = self.model.eval()  # Change model state to evaluation
        self.model.to(self.device)
        if "cuda" in self.device:
            self.model = self.model.to(self.device)
            if self.half:
                self.model.half()

        if self.multiscale:
            self.scales = [0.5, 1.0, 1.5, 2.0]

        coco_result = []
        num_keypoints = Pose.num_kpts

        pbar_eval = None
        if not silent:
            pbar_desc = "Evaluation progress"
            pbar_eval = tqdm(desc=pbar_desc, total=len(data), bar_format="{l_bar}%s{bar}{r_bar}" % '\x1b[38;5;231m')

        for sample in data:
            file_name = sample['file_name']
            img = sample['img']
            h, w, _ = img.shape
            max_width = w
            kernel = int(h / self.first_pass_height)
            if kernel > 0:
                pool_img = self.__pooling(img, kernel)
            else:
                pool_img = img

            avg_pafs = self.__first_pass(pool_img)      # Heatmap Generation
            avg_pafs = avg_pafs.astype(np.float32)

            pafs_map = cv2.blur(avg_pafs, (5, 5))
            pafs_map[pafs_map < self.threshold] = 0

            heatmap = pafs_map.sum(axis=2)
            heatmap = heatmap * 100
            heatmap = heatmap.astype(np.uint8)
            heatmap = cv2.blur(heatmap, (5, 5))

            self.prev_heatmap = heatmap
            heatmap_dims, detection = self.__crop_heatmap(heatmap)

            if detection:
                cropped_heatmap = heatmap[heatmap_dims[2]:heatmap_dims[3], heatmap_dims[0]:heatmap_dims[1]]
                if self.__check_for_split(cropped_heatmap):
                    crops = self.__split_process(cropped_heatmap)   # Split horizontal or vertical

                    crop_part = 0
                    for crop_params in crops:
                        crop = crop_params[0]
                        if crop.size > 0:
                            crop_part += 1

                            xmin, xmax, ymin, ymax = self.__crop_enclosing_bbox(crop)

                            xmin += heatmap_dims[0]
                            xmax += heatmap_dims[0]
                            ymin += heatmap_dims[2]
                            ymax += heatmap_dims[2]

                            xmin += crop_params[1]
                            xmax += crop_params[1]
                            ymin += crop_params[3]
                            ymax += crop_params[3]

                            crop_img, xmin, xmax, ymin, ymax = self.__crop_image_func(xmin, xmax, ymin, ymax, pool_img, img,
                                                                                      heatmap, self.perc)

                            if crop_part == 1:
                                if self.x1min is None:
                                    self.x1min = xmin
                                    self.y1min = ymin
                                    self.x1max = xmax
                                    self.y1max = ymax
                                else:
                                    a = 0.2
                                    self.x1min = a * xmin + (1 - a) * self.x1min
                                    self.y1min = a * ymin + (1 - a) * self.y1min
                                    self.y1max = a * ymax + (1 - a) * self.y1max
                                    self.x1max = a * xmax + (1 - a) * self.x1max

                            elif crop_part == 2:
                                if self.x2min is None:
                                    self.x2min = xmin
                                    self.y2min = ymin
                                    self.x2max = xmax
                                    self.y2max = ymax
                                else:
                                    a = 0.2
                                    self.x2min = a * xmin + (1 - a) * self.x2min
                                    self.y2min = a * ymin + (1 - a) * self.y2min
                                    self.y2max = a * ymax + (1 - a) * self.y2max
                                    self.x2max = a * xmax + (1 - a) * self.x2max

                            h, w, _, = crop_img.shape
                            if h > self.second_pass_height:
                                second_pass_height = self.second_pass_height
                            else:
                                second_pass_height = h

                            # ------- Second pass of the image, inference for pose estimation -------
                            avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, second_pass_height,
                                                                                    max_width, self.stride, upsample_ratio)

                            total_keypoints_num = 0
                            all_keypoints_by_type = []
                            for kpt_idx in range(18):
                                total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                                         total_keypoints_num)

                            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                            for kpt_id in range(all_keypoints.shape[0]):
                                all_keypoints[kpt_id, 0] = ((all_keypoints[kpt_id, 0] *
                                                             self.stride / upsample_ratio - pad[1]) / scale)
                                all_keypoints[kpt_id, 1] = ((all_keypoints[kpt_id, 1] *
                                                             self.stride / upsample_ratio - pad[0]) / scale)

                            for i in range(all_keypoints.shape[0]):
                                for j in range(all_keypoints.shape[1]):
                                    if j == 0:
                                        all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                                    if j == 1:
                                        all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

                            current_poses = []
                            for n in range(len(pose_entries)):
                                if len(pose_entries[n]) == 0:
                                    continue
                                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                                for kpt_id in range(num_keypoints):
                                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                                pose = Pose(pose_keypoints, pose_entries[n][18])
                                current_poses.append(pose)

                            coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

                            image_id = int(file_name[0:file_name.rfind('.')])

                            for idx in range(len(coco_keypoints)):
                                coco_result.append({
                                    'image_id': image_id,
                                    'category_id': 1,  # person
                                    'keypoints': coco_keypoints[idx],
                                    'score': scores[idx]
                                })

                            if self.visualize:
                                for keypoints in coco_keypoints:
                                    for idx in range(len(keypoints) // 3):
                                        cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                                                   3, (255, 0, 255), -1)
                                cv2.imshow('keypoints', img)
                                key = cv2.waitKey()
                                if key == 27:  # esc
                                    return
                else:
                    xmin = heatmap_dims[0]
                    xmax = heatmap_dims[1]
                    ymin = heatmap_dims[2]
                    ymax = heatmap_dims[3]

                    h, w, _ = pool_img.shape
                    xmin = xmin * int((w / heatmap.shape[1])) * kernel
                    xmax = xmax * int((w / heatmap.shape[1])) * kernel
                    ymin = ymin * int((h / heatmap.shape[0])) * kernel
                    ymax = ymax * int((h / heatmap.shape[0])) * kernel

                    extra_pad_x = int(self.perc * (xmax - xmin))
                    extra_pad_y = int(self.perc * (ymax - ymin))

                    if xmin - extra_pad_x > 0:
                        xmin = xmin - extra_pad_x
                    else:
                        xmin = xmin

                    if xmax + extra_pad_x < img.shape[1]:
                        xmax = xmax + extra_pad_x
                    else:
                        xmax = xmax

                    if ymin - extra_pad_y > 0:
                        ymin = ymin - extra_pad_y
                    else:
                        ymin = ymin

                    if ymax + extra_pad_y < img.shape[0]:
                        ymax = ymax + extra_pad_y
                    else:
                        ymax = ymax

                    if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                        crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                    else:
                        crop_img = img[0:img.shape[0], 0:img.shape[1]]

                    h, w, _, = crop_img.shape
                    if h > self.second_pass_height:
                        second_pass_height = self.second_pass_height
                    else:
                        second_pass_height = h

                    # ------- Second pass of the image, inference for pose estimation -------
                    avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, second_pass_height,
                                                                            max_width, self.stride, upsample_ratio)

                    total_keypoints_num = 0
                    all_keypoints_by_type = []
                    for kpt_idx in range(18):
                        total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                                 total_keypoints_num)

                    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                    for kpt_id in range(all_keypoints.shape[0]):
                        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / upsample_ratio - pad[1]) / scale
                        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / upsample_ratio - pad[0]) / scale

                    for i in range(all_keypoints.shape[0]):
                        for j in range(all_keypoints.shape[1]):
                            if j == 0:
                                all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                            if j == 1:
                                all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

                    current_poses = []
                    for n in range(len(pose_entries)):
                        if len(pose_entries[n]) == 0:
                            continue
                        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                        for kpt_id in range(num_keypoints):
                            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                        pose = Pose(pose_keypoints, pose_entries[n][18])
                        current_poses.append(pose)

                    coco_keypoints, scores = convert_to_coco_format(pose_entries, all_keypoints)

                    image_id = int(file_name[0:file_name.rfind('.')])

                    for idx in range(len(coco_keypoints)):
                        coco_result.append({
                            'image_id': image_id,
                            'category_id': 1,  # person
                            'keypoints': coco_keypoints[idx],
                            'score': scores[idx]
                        })

                    if self.visualize:
                        for keypoints in coco_keypoints:
                            for idx in range(len(keypoints) // 3):
                                cv2.circle(img, (int(keypoints[idx * 3]), int(keypoints[idx * 3 + 1])),
                                           3, (255, 0, 255), -1)
                        cv2.imshow('keypoints', img)
                        key = cv2.waitKey()
                        if key == 27:  # esc
                            return

            if not silent:
                pbar_eval.update(1)

        with open(self.output_name, 'w') as f:
            json.dump(coco_result, f, indent=4)
        if len(coco_result) != 0:
            if use_subset:
                result = run_coco_eval(os.path.join(dataset.path, "val_subset.json"),
                                       self.output_name, verbose=not silent)
            else:
                result = run_coco_eval(os.path.join(dataset.path, annotations_filename),
                                       self.output_name, verbose=not silent)
            return {"average_precision": result.stats[0:5], "average_recall": result.stats[5:]}
        else:
            if not silent and verbose:
                print("Evaluation ended with no detections.")
            return {"average_precision": [0.0 for _ in range(5)], "average_recall": [0.0 for _ in range(5)]}

    def infer(self, img, upsample_ratio=4, stride=8, track=True, smooth=True, multiscale=False):
        """
        This method is used to perform pose estimation on an image.

        :param img: image to run inference on
        :rtype img: engine.data.Image class object
        :param upsample_ratio: Defines the amount of upsampling to be performed on the heatmaps and PAFs
            when resizing,defaults to 4
        :type upsample_ratio: int, optional
        :param stride: Defines the stride value for creating a padded image
        :type stride: int,optional
        :param track: If True, infer propagates poses ids from previous frame results to track poses,
            defaults to 'True'
        :type track: bool, optional
        :param smooth: If True, smoothing is performed on pose keypoints between frames, defaults to 'True'
        :type smooth: bool, optional
        :param multiscale: Specifies whether evaluation will run in the predefined multiple scales setup or not.
        :type multiscale: bool,optional

        :return: Returns a list of engine.target.Pose objects, where each holds a pose
        and a heatmap that contains human silhouettes of the input image.
        If no detections were made returns an empty list for poses and a black frame for heatmap.

        :rtype: poses -> list of engine.target.Pose objects
                heatmap -> np.array()
        """
        current_poses = []
        num_keypoints = Pose.num_kpts
        if not isinstance(img, Image):
            img = Image(img)

        # Bring image into the appropriate format for the implementation
        img = img.convert(format='channels_last', channel_order='bgr')
        h, w, _ = img.shape
        max_width = w
        xmin, ymin = 0, 0
        ymax, xmax, _ = img.shape

        if self.counter % 5 == 0:
            kernel = int(h / self.first_pass_height)
            if kernel > 0:
                pool_img = self.__pooling(img, kernel)
            else:
                pool_img = img

            avg_pafs = self.__first_pass(pool_img)      # Heatmap Generation
            avg_pafs = avg_pafs.astype(np.float32)
            pafs_map = cv2.blur(avg_pafs, (5, 5))

            pafs_map[pafs_map < self.threshold] = 0

            heatmap = pafs_map.sum(axis=2)
            heatmap = heatmap * 100
            heatmap = heatmap.astype(np.uint8)
            heatmap = cv2.blur(heatmap, (5, 5))

            self.prev_heatmap = heatmap

            contours, hierarchy = cv2.findContours(heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            count = []

            if len(contours) > 0:
                for x in contours:
                    count.append(x)
                xdim = []
                ydim = []

                for j in range(len(count)):  # Loop for every human (every contour)
                    for i in range(len(count[j])):
                        xdim.append(count[j][i][0][0])
                        ydim.append(count[j][i][0][1])

                h, w, _ = pool_img.shape

                xmin = int(np.floor(min(xdim))) * int((w / heatmap.shape[1])) * kernel
                xmax = int(np.floor(max(xdim))) * int((w / heatmap.shape[1])) * kernel
                ymin = int(np.floor(min(ydim))) * int((h / heatmap.shape[0])) * kernel
                ymax = int(np.floor(max(ydim))) * int((h / heatmap.shape[0])) * kernel

                if self.xmin is None:
                    self.xmin = xmin
                    self.ymin = ymin
                    self.xmax = xmax
                    self.ymax = ymax
                else:
                    a = 0.2
                    self.xmin = a * xmin + (1 - a) * self.xmin
                    self.ymin = a * ymin + (1 - a) * self.ymin
                    self.ymax = a * ymax + (1 - a) * self.ymax
                    self.xmax = a * xmax + (1 - a) * self.xmax

                extra_pad_x = int(self.perc * (self.xmax - self.xmin))  # Adding an extra pad around cropped image
                extra_pad_y = int(self.perc * (self.ymax - self.ymin))

                if self.xmin - extra_pad_x > 0:
                    xmin = self.xmin - extra_pad_x
                else:
                    xmin = self.xmin
                if self.xmax + extra_pad_x < img.shape[1]:
                    xmax = self.xmax + extra_pad_x
                else:
                    xmax = self.xmax

                if self.ymin - extra_pad_y > 0:
                    ymin = self.ymin - extra_pad_y
                else:
                    ymin = self.ymin
                if self.ymax + extra_pad_y < img.shape[0]:
                    ymax = self.ymax + extra_pad_y
                else:
                    ymax = self.ymax

                if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                    crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                else:
                    crop_img = img[0:img.shape[0], 0:img.shape[1]]

                h, w, _ = crop_img.shape
                if crop_img.shape[0] < self.second_pass_height:
                    second_pass_height = crop_img.shape[0]
                else:
                    second_pass_height = self.second_pass_height

                # ------- Second pass of the image, inference for pose estimation -------
                avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, second_pass_height,
                                                                        max_width, self.stride, upsample_ratio)

                total_keypoints_num = 0
                all_keypoints_by_type = []
                for kpt_idx in range(18):
                    total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                             total_keypoints_num)

                pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                for kpt_id in range(all_keypoints.shape[0]):
                    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / upsample_ratio - pad[
                        1]) / scale
                    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / upsample_ratio - pad[
                        0]) / scale

                for i in range(all_keypoints.shape[0]):
                    for j in range(all_keypoints.shape[1]):
                        if j == 0:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                        if j == 1:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

                current_poses = []
                for n in range(len(pose_entries)):
                    if len(pose_entries[n]) == 0:
                        continue
                    pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                    for kpt_id in range(num_keypoints):
                        if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

                    if np.count_nonzero(pose_keypoints == -1) < 26:
                        pose = Pose(pose_keypoints, pose_entries[n][18])
                        current_poses.append(pose)
            else:
                if self.xmin is None:
                    self.xmin = xmin
                    self.ymin = ymin
                    self.xmax = xmax
                    self.ymax = ymax
                else:
                    a = 0.2
                    self.xmin = a * xmin + (1 - a) * self.xmin
                    self.ymin = a * ymin + (1 - a) * self.ymin
                    self.ymax = a * ymax + (1 - a) * self.ymax
                    self.xmax = a * xmax + (1 - a) * self.xmax
        else:

            extra_pad_x = int(self.perc * (self.xmax - self.xmin))  # Adding an extra pad around cropped image
            extra_pad_y = int(self.perc * (self.ymax - self.ymin))

            if self.xmin - extra_pad_x > 0:
                xmin = self.xmin - extra_pad_x
            else:
                xmin = self.xmin

            if self.xmax + extra_pad_x < img.shape[1]:
                xmax = self.xmax + extra_pad_x
            else:
                xmax = self.xmax

            if self.ymin - extra_pad_y > 0:
                ymin = self.ymin - extra_pad_y
            else:
                ymin = self.ymin

            if self.ymax + extra_pad_y < img.shape[0]:
                ymax = self.ymax + extra_pad_y
            else:
                ymax = self.ymax

            if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            else:
                crop_img = img[0:img.shape[0], 0:img.shape[1]]

            h, w, _ = crop_img.shape
            if crop_img.shape[0] < self.second_pass_height:
                second_pass_height = crop_img.shape[0]
            else:
                second_pass_height = self.second_pass_height

            # ------- Second pass of the image, inference for pose estimation -------
            avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, second_pass_height,
                                                                    max_width, self.stride, upsample_ratio)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(18):
                total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                         total_keypoints_num)

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / upsample_ratio - pad[0]) / scale

            for i in range(all_keypoints.shape[0]):
                for j in range(all_keypoints.shape[1]):
                    if j == 0:
                        all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                    if j == 1:
                        all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

            current_poses = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

                if np.count_nonzero(pose_keypoints == -1) < 26:
                    pose = Pose(pose_keypoints, pose_entries[n][18])
                    current_poses.append(pose)

            if np.any(self.prev_heatmap) is False:
                heatmap = np.zeros((int(img.shape[0] / (int((img.shape[0] / self.first_pass_height))) / 8),
                                    int(img.shape[1] / (int((img.shape[0] / self.first_pass_height))) / 8)),
                                   dtype=np.uint8)
            else:
                heatmap = self.prev_heatmap
        self.counter += 1
        bounds = ([self.xmin, self.xmax, self.ymin, self.ymax],)
        return current_poses, heatmap, bounds

    def infer_adaptive(self, img, upsample_ratio=4, stride=8):
        """
            This method is used to perform pose estimation on an image.

            :param img: image to run inference on
            :rtype img: engine.data.Image class object
            :param upsample_ratio: Defines the amount of upsampling to be performed on the heatmaps and PAFs
                when resizing,defaults to 4
            :type upsample_ratio: int, optional
            :param stride: Defines the stride value for creating a padded image
            :type stride: int,optional
            :param track: If True, infer propagates poses ids from previous frame results to track poses,
                defaults to 'True'
            :type track: bool, optional
            :param smooth: If True, smoothing is performed on pose keypoints between frames, defaults to 'True'
            :type smooth: bool, optional
            :param multiscale: Specifies whether evaluation will run in the predefined multiple scales setup or not.
            :type multiscale: bool,optional

            :return: Returns a list of engine.target.Pose objects, where each holds a pose
            and a heatmap that contains human silhouettes of the input image.
            If no detections were made returns an empty list for poses and a black frame for heatmap.

            :rtype: poses -> list of engine.target.Pose objects
                    heatmap -> np.array()
        """
        current_poses = []
        num_keypoints = Pose.num_kpts
        if not isinstance(img, Image):
            img = Image(img)

        # Bring image into the appropriate format for the implementation
        img = img.convert(format='channels_last', channel_order='bgr')
        h, w, _ = img.shape
        max_width = w
        xmin, ymin = 0, 0
        ymax, xmax, _ = img.shape

        if self.counter % 2 == 0:
            kernel = int(h / self.first_pass_height)
            if kernel > 0:
                pool_img = self.__pooling(img, kernel)
            else:
                pool_img = img

            avg_pafs = self.__first_pass(pool_img)      # Heatmap Generation

            avg_pafs = avg_pafs.astype(np.float32)
            pafs_map = cv2.blur(avg_pafs, (5, 5))
            pafs_map[pafs_map < self.threshold] = 0
            heatmap = pafs_map.sum(axis=2)
            heatmap = heatmap * 100
            heatmap = heatmap.astype(np.uint8)
            heatmap = cv2.blur(heatmap, (5, 5))
            self.prev_heatmap = heatmap
            heatmap_dims, detection = self.__crop_heatmap(heatmap)

            if detection:
                self.xmin = heatmap_dims[0] * (img.shape[1] / heatmap.shape[1])
                self.ymin = heatmap_dims[2] * (img.shape[0] / heatmap.shape[0])
                self.xmax = heatmap_dims[1] * (img.shape[1] / heatmap.shape[1])
                self.ymax = heatmap_dims[3] * (img.shape[0] / heatmap.shape[0])
                cropped_heatmap = heatmap[heatmap_dims[2]:heatmap_dims[3], heatmap_dims[0]:heatmap_dims[1]]
                if self.__check_for_split(cropped_heatmap):
                    # Split horizontal or vertical
                    crops = self.__split_process(cropped_heatmap)
                    crop_part = 0
                    for crop_params in crops:
                        crop = crop_params[0]
                        if crop.size > 0:
                            crop_part += 1

                            xmin, xmax, ymin, ymax = self.__crop_enclosing_bbox(crop)

                            xmin += crop_params[1]
                            xmax += crop_params[1]
                            ymin += crop_params[3]
                            ymax += crop_params[3]

                            xmin += heatmap_dims[0]
                            xmax += heatmap_dims[0]
                            ymin += heatmap_dims[2]
                            ymax += heatmap_dims[2]

                            crop_img, xmin, xmax, ymin, ymax = self.__crop_image_func(xmin, xmax, ymin, ymax, pool_img, img,
                                                                                      heatmap, self.perc)

                            if crop_part == 1:
                                if self.x1min is None:
                                    self.x1min = xmin
                                    self.y1min = ymin
                                    self.x1max = xmax
                                    self.y1max = ymax
                                else:
                                    a = 0.2
                                    self.x1min = a * xmin + (1 - a) * self.x1min
                                    self.y1min = a * ymin + (1 - a) * self.y1min
                                    self.y1max = a * ymax + (1 - a) * self.y1max
                                    self.x1max = a * xmax + (1 - a) * self.x1max
                            elif crop_part == 2:
                                if self.x2min is None:
                                    self.x2min = xmin
                                    self.y2min = ymin
                                    self.x2max = xmax
                                    self.y2max = ymax
                                else:
                                    a = 0.2
                                    self.x2min = a * xmin + (1 - a) * self.x2min
                                    self.y2min = a * ymin + (1 - a) * self.y2min
                                    self.y2max = a * ymax + (1 - a) * self.y2max
                                    self.x2max = a * xmax + (1 - a) * self.x2max

                            h, w, _, = crop_img.shape
                            if h > self.second_pass_height:
                                second_pass_height = self.second_pass_height
                            else:
                                second_pass_height = h

                            # ------- Second pass of the image, inference for pose estimation -------
                            avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, second_pass_height,
                                                                                    max_width, self.stride, upsample_ratio)

                            total_keypoints_num = 0
                            all_keypoints_by_type = []
                            for kpt_idx in range(18):
                                total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                                         total_keypoints_num)

                            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                            for kpt_id in range(all_keypoints.shape[0]):
                                all_keypoints[kpt_id, 0] = ((all_keypoints[kpt_id, 0] *
                                                             self.stride / upsample_ratio - pad[1]) / scale)
                                all_keypoints[kpt_id, 1] = ((all_keypoints[kpt_id, 1] *
                                                             self.stride / upsample_ratio - pad[0]) / scale)

                            for i in range(all_keypoints.shape[0]):
                                for j in range(all_keypoints.shape[1]):
                                    if j == 0:
                                        all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                                    if j == 1:
                                        all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

                            for n in range(len(pose_entries)):
                                if len(pose_entries[n]) == 0:
                                    continue
                                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                                for kpt_id in range(num_keypoints):
                                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                                        pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                                        pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                                pose = Pose(pose_keypoints, pose_entries[n][18])
                                current_poses.append(pose)

                else:
                    xmin = heatmap_dims[0]
                    xmax = heatmap_dims[1]
                    ymin = heatmap_dims[2]
                    ymax = heatmap_dims[3]

                    h, w, _ = pool_img.shape
                    xmin = xmin * int((w / heatmap.shape[1])) * kernel
                    xmax = xmax * int((w / heatmap.shape[1])) * kernel
                    ymin = ymin * int((h / heatmap.shape[0])) * kernel
                    ymax = ymax * int((h / heatmap.shape[0])) * kernel

                    extra_pad_x = int(self.perc * (xmax - xmin))
                    extra_pad_y = int(self.perc * (ymax - ymin))

                    if xmin - extra_pad_x > 0:
                        xmin = xmin - extra_pad_x
                    else:
                        xmin = xmin

                    if xmax + extra_pad_x < img.shape[1]:
                        xmax = xmax + extra_pad_x
                    else:
                        xmax = xmax

                    if ymin - extra_pad_y > 0:
                        ymin = ymin - extra_pad_y
                    else:
                        ymin = ymin

                    if ymax + extra_pad_y < img.shape[0]:
                        ymax = ymax + extra_pad_y
                    else:
                        ymax = ymax

                    if self.xmin is None:
                        self.xmin = xmin
                        self.ymin = ymin
                        self.xmax = xmax
                        self.ymax = ymax
                        self.x1min, self.x1max, self.y1min, self.y1max = xmin, xmax, ymin, ymax
                        self.x2min, self.x2max, self.y2min, self.y2max = xmin, xmax, ymin, ymax
                    else:
                        a = 0.2
                        self.xmin = a * xmin + (1 - a) * self.xmin
                        self.ymin = a * ymin + (1 - a) * self.ymin
                        self.ymax = a * ymax + (1 - a) * self.ymax
                        self.xmax = a * xmax + (1 - a) * self.xmax
                        self.x1min, self.x1max, self.y1min, self.y1max = self.xmin, self.xmax, self.ymin, self.ymax
                        self.x2min, self.x2max, self.y2min, self.y2max = self.xmin, self.xmax, self.ymin, self.ymax

                    if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                        crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                    else:
                        crop_img = img[0:img.shape[0], 0:img.shape[1]]

                    h, w, _, = crop_img.shape
                    if h > self.second_pass_height:
                        second_pass_height = self.second_pass_height
                    else:
                        second_pass_height = h

                    # ------- Second pass of the image, inference for pose estimation -------
                    avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, second_pass_height,
                                                                            max_width, self.stride, upsample_ratio)

                    total_keypoints_num = 0
                    all_keypoints_by_type = []
                    for kpt_idx in range(18):
                        total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                                 total_keypoints_num)

                    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                    for kpt_id in range(all_keypoints.shape[0]):
                        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

                    for i in range(all_keypoints.shape[0]):
                        for j in range(all_keypoints.shape[1]):
                            if j == 0:
                                all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                            if j == 1:
                                all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

                    current_poses = []
                    for n in range(len(pose_entries)):
                        if len(pose_entries[n]) == 0:
                            continue
                        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                        for kpt_id in range(num_keypoints):
                            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                        pose = Pose(pose_keypoints, pose_entries[n][18])
                        current_poses.append(pose)
            else:
                if self.xmin is None:
                    self.xmin = xmin
                    self.ymin = ymin
                    self.xmax = xmax
                    self.ymax = ymax
                    self.x1min, self.x1max, self.y1min, self.y1max = 0, 0, 0, 0
                    self.x2min, self.x2max, self.y2min, self.y2max = 0, 0, 0, 0
                else:
                    a = 0.8
                    self.xmin = a * xmin + (1 - a) * self.xmin
                    self.ymin = a * ymin + (1 - a) * self.ymin
                    self.ymax = a * ymax + (1 - a) * self.ymax
                    self.xmax = a * xmax + (1 - a) * self.xmax

                    self.x1min, self.x1max, self.y1min, self.y1max = 0, 0, 0, 0
                    self.x2min, self.x2max, self.y2min, self.y2max = 0, 0, 0, 0
        else:
            if self.x1min is None:
                self.x1min = xmin
                self.y1min = ymin
                self.x1max = xmax
                self.y1max = ymax
            if self.x2min is None:
                self.x2min = xmin
                self.y2min = ymin
                self.x2max = xmax
                self.y2max = ymax

            boxes = ([self.x1min, self.x1max, self.y1min, self.y1max], [self.x2min, self.x2max, self.y2min, self.y2max])
            for box in boxes:
                xmin = box[0]
                xmax = box[1]
                ymin = box[2]
                ymax = box[3]

                extra_pad_x = int(self.perc * (xmax - xmin))
                extra_pad_y = int(self.perc * (ymax - ymin))

                if (xmin - extra_pad_x > 0) and (xmin > 0):
                    xmin = xmin - extra_pad_x
                else:
                    xmin = xmin

                if (xmax + extra_pad_x < img.shape[1]) and (xmax < img.shape[1]):
                    xmax = xmax + extra_pad_x
                else:
                    xmax = xmax

                if (ymin - extra_pad_y > 0) and (ymin > 0):
                    ymin = ymin - extra_pad_y
                else:
                    ymin = ymin

                if (ymax + extra_pad_y < img.shape[0]) and (ymax < img.shape[0]):
                    ymax = ymax + extra_pad_y
                else:
                    ymax = ymax

                if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                    crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
                else:
                    crop_img = img[0:img.shape[0], 0:img.shape[1]]

                h, w, _ = crop_img.shape
                if crop_img.shape[0] < self.second_pass_height:
                    second_pass_height = crop_img.shape[0]
                else:
                    second_pass_height = self.second_pass_height

                # ------- Second pass of the image, inference for pose estimation -------
                avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, second_pass_height,
                                                                        max_width, self.stride, upsample_ratio)

                total_keypoints_num = 0
                all_keypoints_by_type = []
                for kpt_idx in range(18):
                    total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                             total_keypoints_num)

                pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                for kpt_id in range(all_keypoints.shape[0]):
                    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / upsample_ratio - pad[1]) / scale
                    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / upsample_ratio - pad[0]) / scale

                for i in range(all_keypoints.shape[0]):
                    for j in range(all_keypoints.shape[1]):
                        if j == 0:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + xmin))
                        if j == 1:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + ymin))

                current_poses = []
                for n in range(len(pose_entries)):
                    if len(pose_entries[n]) == 0:
                        continue
                    pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
                    for kpt_id in range(num_keypoints):
                        if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

                    if np.count_nonzero(pose_keypoints == -1) < 26:
                        pose = Pose(pose_keypoints, pose_entries[n][18])
                        current_poses.append(pose)

                if np.any(self.prev_heatmap) is False:
                    heatmap = np.zeros((int(img.shape[0] / (int((img.shape[0] / self.first_pass_height))) / 8),
                                        int(img.shape[1] / (int((img.shape[0] / self.first_pass_height))) / 8)),
                                       dtype=np.uint8)
                else:
                    heatmap = self.prev_heatmap
        self.counter += 1

        bounds = [(self.x1min, self.x1max, self.y1min, self.y1max),
                  (self.x2min, self.x2max, self.y2min, self.y2max)]
        return current_poses, heatmap, bounds

    def download(self, path=None, mode="pretrained", verbose=False,
                 url=OPENDR_SERVER_URL + "perception/pose_estimation/lightweight_open_pose/",
                 image_resolution=1080):
        """
        Download utility for various Lightweight Open Pose components. Downloads files depending on mode and
        saves them in the path provided. It supports downloading:
        1) the default mobilenet pretrained model
        2) mobilenet, mobilenetv2 and shufflenet weights needed for training
        3) a test dataset with a single COCO image and its annotation
        :param path: Local path to save the files, defaults to self.temp_path if None
        :type path: str, path, optional
        :param mode: What file to download, can be one of "pretrained", "weights", "test_data", defaults to "pretrained"
        :type mode: str, optional
        :param verbose: Whether to print messages in the console, defaults to False
        :type verbose: bool, optional
        :param url: URL of the FTP server, defaults to OpenDR FTP URL
        :type url: str, optional
        :param image_resolution: Resolution of the test images to download
        :type image_resolution: int, optional
        """
        valid_modes = ["weights", "pretrained", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode in ("pretrained", "weights"):
            super(HighResolutionPoseEstimationLearner, self).download(path=path, mode=mode, verbose=verbose, url=url)
        elif mode == "test_data":
            if verbose:
                print("Downloading test data...")
            if not os.path.exists(os.path.join(self.temp_path, "dataset")):
                os.makedirs(os.path.join(self.temp_path, "dataset"))
            if not os.path.exists(os.path.join(self.temp_path, "dataset", "image")):
                os.makedirs(os.path.join(self.temp_path, "dataset", "image"))
            # Path for high resolution data
            url = OPENDR_SERVER_URL + "perception/pose_estimation/high_resolution_pose_estimation/"
            # Download annotation file
            file_url = os.path.join(url, "dataset", "annotation.json")
            urlretrieve(file_url, os.path.join(self.temp_path, "dataset", "annotation.json"))
            # Download test image
            if image_resolution in (1080, 1440):
                file_url = os.path.join(url, "dataset", "image", "000000000785_" + str(image_resolution) + ".jpg")
                urlretrieve(file_url, os.path.join(self.temp_path, "dataset", "image", "000000000785_1080.jpg"))
                file_url = os.path.join(url, "dataset", "image", "000000052591_" + str(image_resolution) + ".jpg")
                urlretrieve(file_url, os.path.join(self.temp_path, "dataset", "image", "000000052591_1080.jpg"))
            else:
                raise UserWarning("There are no data for this image resolution (only 1080 and 1440 are supported).")

            if verbose:
                print("Test data download complete.")

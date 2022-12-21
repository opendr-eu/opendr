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
# limitations under the License."""

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
                 first_pass_height=360, second_pass_height=540, percentage_arround_crop=0.3, heatmap_threshold=0.1,
                 experiment_name='default', num_workers=8, weights_only=True, output_name='detections.json',
                 multiscale=False, scales=None, visualize=False,
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
        self.perc = percentage_arround_crop
        self.threshold = heatmap_threshold

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

         :returns: the heatmap of human figures, the part affinity filed (pafs), the scale of the resized image compred
            to the initial and the pad arround the image
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

    @staticmethod
    def __pooling(img, kernel):  # Pooling on input image for dimension reduction
        """This method applies a pooling filter on an input image in order to resize it in a fixed shape

         :param img: input image for resizing
         :rtype img: engine.data.Image class object
         :param kernel: the kernel size of the pooling filter
         :type kernel: int
         """
        pool_img = torchvision.transforms.ToTensor()(img)
        pool_img = pool_img.unsqueeze(0)
        pool_img = torch.nn.functional.avg_pool2d(pool_img, kernel)
        pool_img = pool_img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return pool_img

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

        img_height = data[0]['img'].shape[0]

        if img_height in (1080, 1440):
            offset = 200
        elif img_height == 720:
            offset = 50
        else:
            offset = 0

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

            # ------- Heatmap Generation -------
            avg_pafs = self.__first_pass(pool_img)
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
                    crop_img = img[offset:img.shape[0], offset:img.shape[1]]

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
                        if j == 0:  # Adjust offset if needed for evaluation on our HR datasets
                            all_keypoints[i][j] = round((all_keypoints[i][j] + xmin) - offset)
                        if j == 1:  # Adjust offset if needed for evaluation on our HR datasets
                            all_keypoints[i][j] = round((all_keypoints[i][j] + ymin) - offset)

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
                        cv2.circle(img, (int(keypoints[idx * 3] + offset), int(keypoints[idx * 3 + 1]) + offset),
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

    def infer(self, img, upsample_ratio=4, stride=8, track=True, smooth=True,
              multiscale=False):
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

                :return: Returns a list of engine.target.Pose objects, where each holds a pose, or returns an empty list
                    if no detections were made.
                :rtype: list of engine.target.Pose objects
                """
        current_poses = []

        offset = 0

        num_keypoints = Pose.num_kpts

        if not isinstance(img, Image):
            img = Image(img)

        # Bring image into the appropriate format for the implementation
        img = img.convert(format='channels_last', channel_order='bgr')

        h, w, _ = img.shape
        max_width = w

        kernel = int(h / self.first_pass_height)
        if kernel > 0:
            pool_img = self.__pooling(img, kernel)
        else:
            pool_img = img

        # ------- Heatmap Generation -------
        avg_pafs = self.__first_pass(pool_img)
        avg_pafs = avg_pafs.astype(np.float32)
        pafs_map = cv2.blur(avg_pafs, (5, 5))

        pafs_map[pafs_map < self.threshold] = 0

        heatmap = pafs_map.sum(axis=2)
        heatmap = heatmap * 100
        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.blur(heatmap, (5, 5))

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
                crop_img = img[offset:img.shape[0], offset:img.shape[1]]

            h, w, _ = crop_img.shape

            # ------- Second pass of the image, inference for pose estimation -------
            avg_heatmaps, avg_pafs, scale, pad = self.__second_pass(crop_img, self.second_pass_height,
                                                                    max_width, stride, upsample_ratio)

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
                    if j == 0:  # Adjust offset if needed for evaluation on our HR datasets
                        all_keypoints[i][j] = round((all_keypoints[i][j] + xmin) - offset)
                    if j == 1:  # Adjust offset if needed for evaluation on our HR datasets
                        all_keypoints[i][j] = round((all_keypoints[i][j] + ymin) - offset)

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

        return current_poses

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
            else:
                raise UserWarning("There are no data for this image resolution (only 1080 and 1440 are supported).")

            if verbose:
                print("Test data download complete.")

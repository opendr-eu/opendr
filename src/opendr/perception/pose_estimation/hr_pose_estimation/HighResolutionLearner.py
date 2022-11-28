# General imports
# General imports
import torchvision.transforms
import onnxruntime as ort
import os
import ntpath
import shutil
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm

from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.engine.data import Image
from opendr.engine.target import Pose
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR lightweight_open_pose imports

from opendr.perception.pose_estimation.hr_pose_estimation.filtered_pose import FilteredPose
from opendr.perception.pose_estimation.hr_pose_estimation.utilities import track_poses
from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.models.with_mobilenet import \
    PoseEstimationWithMobileNet
from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.models.with_mobilenet_v2 import \
    PoseEstimationWithMobileNetV2
from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.models.with_shufflenet import \
    PoseEstimationWithShuffleNet

from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.modules.load_state import \
    load_state  # , load_from_mobilenet
from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.modules.keypoints import \
    extract_keypoints, group_keypoints
from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.datasets.coco import CocoValDataset

from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.val import \
    convert_to_coco_format, run_coco_eval, normalize, pad_width
from opendr.perception.pose_estimation.hr_pose_estimation.algorithm.scripts import \
    make_val_subset


class HighResolutionPoseEstimationLearner(Learner):

    def __init__(self, device='cuda', backbone='mobilenet',
                 temp_path='temp', mobilenet_use_stride=True, mobilenetv2_width=1.0, shufflenet_groups=3,
                 num_refinement_stages=2, batches_per_iter=1,base_height=256,
                 experiment_name='default', num_workers=8, weights_only=True, output_name='detections.json',
                 multiscale=False, scales=None, visualize=False,
                 img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256), pad_value=(0, 0, 0),
                 half_precision=False):

        super(HighResolutionPoseEstimationLearner, self).__init__(temp_path=temp_path, device=device, backbone=backbone)
        self.parent_dir = temp_path  # Parent dir should be filled by the user according to README

        self.num_refinement_stages = num_refinement_stages  # How many extra refinement stages to add
        self.batches_per_iter = batches_per_iter

        self.experiment_name = experiment_name
        self.num_workers = num_workers
        self.backbone = backbone.lower()
        self.half = half_precision

        supportedBackbones = ["mobilenet", "mobilenetv2", "shufflenet"]
        if self.backbone not in supportedBackbones:
            raise ValueError(self.backbone + " not a valid backbone. Supported backbones:" + str(supportedBackbones))
        if self.backbone == "mobilenet":
            self.use_stride = mobilenet_use_stride
        else:
            self.use_stride = None
        if self.backbone == "mobilenetv2":
            self.mobilenetv2_width = mobilenetv2_width
        if self.backbone == "shufflenet":
            self.shufflenet_groups = shufflenet_groups
        # if self.backbone == "mobilenet":
        #     self.from_mobilenet = True # TODO from_mobilenet = True, bugs out the loading
        # else:
        #     self.from_mobilenet = False

        self.weights_only = weights_only  # If True, it won't load optimizer, scheduler, num_iter, current_epoch

        self.output_name = os.path.join(self.parent_dir, output_name)  # Path to json file containing detections
        self.visualize = visualize
        self.base_height = base_height
        if scales is None:
            scales = [1]
        self.multiscale = multiscale  # If set to true, overwrites self.scales to [0.5, 1.0, 1.5, 2.0]
        self.scales = scales
        if self.use_stride:
            self.stride = 8 * 2
        else:
            self.stride = 8
        self.upsample_ratio = 4

        self.img_mean = img_mean
        self.img_scale = img_scale
        self.pad_value = pad_value
        self.previous_poses = []

        self.ort_session = None  # ONNX runtime inference session
        self.model_train_state = True

    def first_pass(self, net, img):
        height, width, _ = img.shape

        if 'cuda' in self.device:
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            tensor_img = tensor_img.cuda()
            if self.half:
                tensor_img = tensor_img.half()
        else:
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cpu()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        stages_output = net(tensor_img)

        end.record()
        torch.cuda.synchronize()

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))

        avg_pafs = pafs
        return avg_pafs

    def second_pass_infer(self, net, img, net_input_height_size, max_width, stride, upsample_ratio, device,
                          pad_value=(0, 0, 0),
                          img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256)):
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

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        stages_output = net(tensor_img)

        end.record()
        torch.cuda.synchronize()

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = heatmaps.astype(np.float32)
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = pafs.astype(np.float32)
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def Pooling(self, img, kernel):  # Pooling on input image for dim reduction
        pooling = torch.nn.AvgPool2d(kernel)
        pool_img = torchvision.transforms.ToTensor()(img)
        pool_img = pool_img.unsqueeze(0)
        pool_img = pooling(pool_img)
        pool_img = pool_img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        return pool_img

    def fit(self, val_dataset=None, logging_path='', logging_flush_secs=30,
            silent=False, verbose=True, use_val_subset=True, val_subset_size=250,
            val_images_folder_name="val2017", val_annotations_filename="person_keypoints_val2017.json"):
        raise NotImplementedError

    def optimize(self, do_constant_folding=False):
        raise NotImplementedError


    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def save(self, path, verbose=False):
        """
        This method is used to save a trained model.
        Provided with the path, absolute or relative, including a *folder* name, it creates a directory with the name
        of the *folder* provided and saves the model inside with a proper format and a .json file with metadata.

        If self.optimize was ran previously, it saves the optimized ONNX model in a similar fashion, by copying it
        from the self.temp_path it was saved previously during conversion.

        :param path: for the model to be saved, including the folder name
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        if self.model is None and self.ort_session is None:
            raise UserWarning("No model is loaded, cannot save.")

        folder_name, _, tail = self.__extract_trailing(path)  # Extract trailing folder name from path
        # Also extract folder name without any extension if extension is erroneously provided
        folder_name_no_ext = folder_name.split(sep='.')[0]

        # Extract path without folder name, by removing folder name from original path
        path_no_folder_name = path.replace(folder_name, '')
        # If tail is '', then path was a/b/c/, which leaves a trailing double '/'
        if tail == '':
            path_no_folder_name = path_no_folder_name[0:-1]  # Remove one '/'

        # Create model directory
        full_path_to_model_folder = path_no_folder_name + folder_name_no_ext
        os.makedirs(full_path_to_model_folder, exist_ok=True)

        model_metadata = {"model_paths": [], "framework": "pytorch", "format": "", "has_data": False,
                          "inference_params": {}, "optimized": None, "optimizer_info": {}, "backbone": self.backbone}

        if self.ort_session is None:
            model_metadata["model_paths"] = [folder_name_no_ext + ".pth"]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pth"

            custom_dict = {'state_dict': self.model.state_dict()}
            torch.save(custom_dict, os.path.join(full_path_to_model_folder, model_metadata["model_paths"][0]))
            if verbose:
                print("Saved Pytorch model.")
        else:
            model_metadata["model_paths"] = [os.path.join(folder_name_no_ext + ".onnx")]
            model_metadata["optimized"] = True
            model_metadata["format"] = "onnx"
            # Copy already optimized model from temp path
            shutil.copy2(os.path.join(self.temp_path, "onnx_model_temp.onnx"),
                         os.path.join(full_path_to_model_folder, model_metadata["model_paths"][0]))
            model_metadata["optimized"] = True
            if verbose:
                print("Saved ONNX model.")

        with open(os.path.join(full_path_to_model_folder, folder_name_no_ext + ".json"), 'w') as outfile:
            json.dump(model_metadata, outfile)

    def eval(self, dataset, first_pass_height, second_pass_height, silent=False, verbose=True, use_subset=True,
             subset_size=250,
             images_folder_name="val2017", annotations_filename="person_keypoints_val2017.json"):

        data = self.__prepare_val_dataset(dataset, use_subset=use_subset,
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

            # Loads weights in self.model from checkpoint
            # if self.from_mobilenet:  # TODO see todo on ctor
            #     load_from_mobilenet(self.model, checkpoint)
            # else:
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

        self.net = self.model
        coco_result = []
        num_keypoints = Pose.num_kpts

        pbar_eval = None
        if not silent:
            pbarDesc = "Evaluation progress"
            pbar_eval = tqdm(desc=pbarDesc, total=len(data), bar_format="{l_bar}%s{bar}{r_bar}" % '\x1b[38;5;231m')

        img_height = data[0]['img'].shape[0]

        if img_height == 1080 or img_height == 1440:
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
            kernel = int(h / first_pass_height)
            if kernel > 0:
                pool_img = HighResolutionPoseEstimationLearner.Pooling(self, img, kernel)
                base_height = pool_img.shape[0]
            else:
                pool_img = img
                base_height = img.shape[0]

            perc = 0.3  # percentage around cropping

            thresshold = 0.1  # thresshold for heatmap

            # ------- Heatmap Generation -------
            avg_pafs = HighResolutionPoseEstimationLearner.first_pass(self, self.net, pool_img)
            avg_pafs = avg_pafs.astype(np.float32)

            pafs_map = cv2.blur(avg_pafs, (5, 5))
            pafs_map[pafs_map < thresshold] = 0

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

                extra_pad_x = int(perc * (xmax - xmin))  # Adding an extra pad around cropped image
                extra_pad_y = int(perc * (ymax - ymin))

                if xmin - extra_pad_x > 0: xmin = xmin - extra_pad_x
                if xmax + extra_pad_x < img.shape[1]: xmax = xmax + extra_pad_x
                if ymin - extra_pad_y > 0: ymin = ymin - extra_pad_y
                if ymax + extra_pad_y < img.shape[0]: ymax = ymax + extra_pad_y

                if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                    crop_img = img[ymin:ymax, xmin:xmax]
                else:
                    crop_img = img[offset:img.shape[0], offset:img.shape[1]]

                h, w, _ = crop_img.shape

                # ------- Second pass of the image, inference for pose estimation -------
                avg_heatmaps, avg_pafs, scale, pad = HighResolutionPoseEstimationLearner.second_pass_infer(self, self.net,
                                                                                                           crop_img,
                                                                                                           second_pass_height,
                                                                                                           max_width,
                                                                                                           self.stride,
                                                                                                           self.upsample_ratio,
                                                                                                           self.device)
                total_keypoints_num = 0
                all_keypoints_by_type = []
                for kpt_idx in range(18):
                    total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                             total_keypoints_num)

                pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)

                for kpt_id in range(all_keypoints.shape[0]):
                    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - pad[
                        1]) / scale
                    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - pad[
                        0]) / scale

                for i in range(all_keypoints.shape[0]):
                    for j in range(all_keypoints.shape[1]):
                        if j == 0:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + xmin) - offset)  # Adjust offset if needed for evaluation on our HR datasets
                        if j == 1:
                            all_keypoints[i][j] = round((all_keypoints[i][j] + ymin) - offset)  # Adjust offset if needed for evaluation on our HR datasets

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
                        'score': scores[idx],
                        'person': idx
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

    def infer(self, img, first_pass_height, second_pass_height, upsample_ratio=4, stride=8, track=True, smooth=True,
              multiscale=False, visualize=False):
        self.net = self.model
        current_poses = []

        offset = 0
        num_keypoints = Pose.num_kpts

        if not isinstance(img, Image):
            img = Image(img)

            # Bring image into the appropriate format for the implementation
        img = img.convert(format='channels_last', channel_order='bgr')

        h, w, _ = img.shape
        max_width = w
        kernel = int(h / first_pass_height)
        if kernel > 0:
            pool_img = HighResolutionPoseEstimationLearner.Pooling(self, img, kernel)
            base_height = pool_img.shape[0]
        else:
            pool_img = img
            base_height = img.shape[0]

        perc = 0.3  # percentage around cropping

        thresshold = 0.1  # threshold for heatmap

        # ------- Heatmap Generation -------
        avg_pafs = HighResolutionPoseEstimationLearner.first_pass(self, self.net, pool_img)
        avg_pafs = avg_pafs.astype(np.float32)
        pafs_map = cv2.blur(avg_pafs, (5, 5))

        pafs_map[pafs_map < thresshold] = 0

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

            extra_pad_x = int(perc * (xmax - xmin))  # Adding an extra pad around cropped image
            extra_pad_y = int(perc * (ymax - ymin))

            if xmin - extra_pad_x > 0: xmin = xmin - extra_pad_x
            if xmax + extra_pad_x < img.shape[1]: xmax = xmax + extra_pad_x
            if ymin - extra_pad_y > 0: ymin = ymin - extra_pad_y
            if ymax + extra_pad_y < img.shape[0]: ymax = ymax + extra_pad_y

            if (xmax - xmin) > 40 and (ymax - ymin) > 40:
                crop_img = img[ymin:ymax, xmin:xmax]
            else:
                crop_img = img[offset:img.shape[0], offset:img.shape[1]]

            h, w, _ = crop_img.shape

            # ------- Second pass of the image, inference for pose estimation -------
            avg_heatmaps, avg_pafs, scale, pad = HighResolutionPoseEstimationLearner.second_pass_infer(self, self.net,
                                                                                                       crop_img,
                                                                                                       second_pass_height,
                                                                                                       max_width,
                                                                                                       stride,
                                                                                                       upsample_ratio,
                                                                                                       self.device)
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
                        all_keypoints[i][j] = round((all_keypoints[i][j] + xmin) - offset)  # Adjust offset if needed for evaluation on our HR datasets
                    if j == 1:
                        all_keypoints[i][j] = round((all_keypoints[i][j] + ymin) - offset)  # Adjust offset if needed for evaluation on our HR datasets

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

    def infer_light_odr(self, img, upsample_ratio=4, track=True, smooth=True):      #LwOP from OpenDR implementation
        """
        This method is used to perform pose estimation on an image.

        :param img: image to run inference on
        :rtype img: engine.data.Image class object
        :param upsample_ratio: Defines the amount of upsampling to be performed on the heatmaps and PAFs when resizing,
            defaults to 4
        :type upsample_ratio: int, optional
        :param track: If True, infer propagates poses ids from previous frame results to track poses, defaults to 'True'
        :type track: bool, optional
        :param smooth: If True, smoothing is performed on pose keypoints between frames, defaults to 'True'
        :type smooth: bool, optional
        :return: Returns a list of engine.target.Pose objects, where each holds a pose, or returns an empty list if no
            detections were made.
        :rtype: list of engine.target.Pose objects
        """
        if not isinstance(img, Image):
            img = Image(img)

        # Bring image into the appropriate format for the implementation
        img = img.convert(format='channels_last', channel_order='bgr')

        height, width, _ = img.shape
        scale = self.base_height / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, self.img_mean, self.img_scale)
        min_dims = [self.base_height, max(scaled_img.shape[1], self.base_height)]
        padded_img, pad = pad_width(scaled_img, self.stride, self.pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if "cuda" in self.device:
            tensor_img = tensor_img.to(self.device)
            if self.half:
                tensor_img = tensor_img.half()

        if self.ort_session is not None:
            stages_output = self.ort_session.run(None, {'data': np.array(tensor_img.cpu())})
            stage2_heatmaps = torch.tensor(stages_output[-2])
            stage2_pafs = torch.tensor(stages_output[-1])
        else:
            if self.model is None:
                raise UserWarning("No model is loaded, cannot run inference. Load a model first using load().")
            if self.model_train_state:
                self.model.eval()
                self.model_train_state = False
            stages_output = self.model(tensor_img)
            stage2_heatmaps = stages_output[-2]
            stage2_pafs = stages_output[-1]

        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        if self.half:
            heatmaps = np.float32(heatmaps)
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        if self.half:
            pafs = np.float32(pafs)
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        num_keypoints = 18
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                     total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / upsample_ratio - pad[0]) / scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            if smooth:
                pose = FilteredPose(pose_keypoints, pose_entries[n][18])
            else:
                pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if track:
            track_poses(self.previous_poses, current_poses, smooth=smooth)
            self.previous_poses = current_poses
        return current_poses

    def init_model(self):
        if self.model is None:
            # No model loaded, initializing new
            if self.backbone == "mobilenet":
                self.model = PoseEstimationWithMobileNet(self.num_refinement_stages, use_stride=self.use_stride)
            elif self.backbone == "mobilenetv2":
                self.model = PoseEstimationWithMobileNetV2(self.num_refinement_stages,
                                                           width_mult=self.mobilenetv2_width)
            elif self.backbone == "shufflenet":
                self.model = PoseEstimationWithShuffleNet(self.num_refinement_stages,
                                                          groups=self.shufflenet_groups)
        else:
            raise UserWarning("Tried to initialize model while model is already initialized.")
        self.model.to(self.device)

    def load(self, path, verbose=False):
        """
        Loads the model from inside the path provided, based on the metadata .json file included.
        :param path: path of the directory the model was saved
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        model_name, _, _ = self.__extract_trailing(path)  # Trailing folder name from the path provided

        with open(os.path.join(path, model_name + ".json")) as metadata_file:
            metadata = json.load(metadata_file)

        self.backbone = metadata["backbone"]
        if not metadata["optimized"]:
            self.__load_from_pth(os.path.join(path, metadata['model_paths'][0]))
            if verbose:
                print("Loaded Pytorch model.")
        else:
            self.__load_from_onnx(os.path.join(path, metadata['model_paths'][0]))
            if verbose:
                print("Loaded ONNX model.")

    def __load_from_pth(self, path):
        """
        This method loads a regular Pytorch model from the path provided into self.model.

        :param path: path to .pth model
        :type path: str
        """
        self.init_model()
        checkpoint = torch.load(path, map_location=torch.device(self.device))
        # if self.from_mobilenet:  # TODO see todo on ctor
        #     load_from_mobilenet(self.model, checkpoint)
        # else:
        load_state(self.model, checkpoint)
        if "cuda" in self.device:
            self.model.to(self.device)
            if self.half:
                self.model.half()
        self.model.train(False)

    def __load_from_onnx(self, path):
        """
        This method loads an ONNX model from the path provided into an onnxruntime inference session.

        :param path: path to ONNX model
        :type path: str
        """
        self.ort_session = ort.InferenceSession(path)

        # The comments below are the alternative way to use the onnx model, it might be useful in the future
        # depending on how ONNX saving/loading will be implemented across the toolkit.
        # # Load the ONNX model
        # self.model = onnx.load(path)
        #
        # # Check that the IR is well formed
        # onnx.checker.check_model(self.model)
        #
        # # Print a human readable representation of the graph
        # onnx.helper.printable_graph(self.model.graph)

    @staticmethod
    def __extract_trailing(path):
        """
        Extracts the trailing folder name or filename from a path provided in an OS-generic way, also handling
        cases where the last trailing character is a separator. Returns the folder name and the split head and tail.

        :param path: the path to extract the trailing filename or folder name from
        :type path: str
        :return: the folder name, the head and tail of the path
        :rtype: tuple of three strings
        """
        head, tail = ntpath.split(path)
        folder_name = tail or ntpath.basename(head)  # handle both a/b/c and a/b/c/
        return folder_name, head, tail

    def download(self, path=None, mode="pretrained", verbose=False,
                 url=OPENDR_SERVER_URL + "perception/pose_estimation/lightweight_open_pose/"):
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
        """
        valid_modes = ["weights", "pretrained", "test_data"]
        if mode not in valid_modes:
            raise UserWarning("mode parameter not valid:", mode, ", file should be one of:", valid_modes)

        if path is None:
            path = self.temp_path

        if not os.path.exists(path):
            os.makedirs(path)

        if mode == "pretrained":
            # Create model's folder
            path = os.path.join(path, "openpose_default")
            if not os.path.exists(path):
                os.makedirs(path)

            if verbose:
                print("Downloading pretrained model...")

            # Download the model's files
            if self.backbone == "mobilenet":
                if not os.path.exists(os.path.join(path, "openpose_default.json")):
                    file_url = os.path.join(url, "openpose_default/openpose_default.json")
                    urlretrieve(file_url, os.path.join(path, "openpose_default.json"))
                    if verbose:
                        print("Downloaded metadata json.")
                else:
                    if verbose:
                        print("Metadata json file already exists.")
                if not os.path.exists(os.path.join(path, "openpose_default.pth")):
                    file_url = os.path.join(url, "openpose_default/openpose_default.pth")
                    urlretrieve(file_url, os.path.join(path, "openpose_default.pth"))
                else:
                    if verbose:
                        print("Trained model .pth file already exists.")
            elif self.backbone == "mobilenetv2":
                raise UserWarning("mobilenetv2 does not support pretrained model.")
            elif self.backbone == "shufflenet":
                raise UserWarning("shufflenet does not support pretrained model.")
            if verbose:
                print("Pretrained model download complete.")

        elif mode == "weights":
            if verbose:
                print("Downloading weights file...")
            if self.backbone == "mobilenet":
                if not os.path.exists(os.path.join(self.temp_path, "mobilenet_sgd_68.848.pth.tar")):
                    file_url = os.path.join(url, "mobilenet_sgd_68.848.pth.tar")
                    urlretrieve(file_url, os.path.join(self.temp_path, "mobilenet_sgd_68.848.pth.tar"))
                    if verbose:
                        print("Downloaded mobilenet weights.")
                else:
                    if verbose:
                        print("Weights file already exists.")
            elif self.backbone == "mobilenetv2":
                if not os.path.exists(os.path.join(self.temp_path, "mobilenetv2_1.0-f2a8633.pth.tar")):
                    file_url = os.path.join(url, "mobilenetv2_1.0-f2a8633.pth.tar")
                    urlretrieve(file_url, os.path.join(self.temp_path, "mobilenetv2_1.0-f2a8633.pth.tar"))
                    if verbose:
                        print("Downloaded mobilenetv2 weights.")
                else:
                    if verbose:
                        print("Weights file already exists.")
            elif self.backbone == "shufflenet":
                if not os.path.exists(os.path.join(self.temp_path, "shufflenet.pth.tar")):
                    file_url = os.path.join(url, "shufflenet.pth.tar")
                    urlretrieve(file_url, os.path.join(self.temp_path, "shufflenet.pth.tar"))
                    if verbose:
                        print("Downloaded shufflenet weights.")
                else:
                    if verbose:
                        print("Weights file already exists.")
            if verbose:
                print("Weights file download complete.")

        elif mode == "test_data":
            if verbose:
                print("Downloading test data...")
            if not os.path.exists(os.path.join(self.temp_path, "dataset")):
                os.makedirs(os.path.join(self.temp_path, "dataset"))
            if not os.path.exists(os.path.join(self.temp_path, "dataset", "image")):
                os.makedirs(os.path.join(self.temp_path, "dataset", "image"))
            # Download annotation file
            file_url = os.path.join(url, "dataset", "annotation.json")
            urlretrieve(file_url, os.path.join(self.temp_path, "dataset", "annotation.json"))
            # Download test image
            url=OPENDR_SERVER_URL + "/perception/pose_estimation/high_resolution_pose_estimation/"
            file_url = os.path.join(url, "dataset", "image", "000000000785_1080.jpg")
            urlretrieve(file_url, os.path.join(self.temp_path, "dataset", "image", "000000000785_1080.jpg"))

            if verbose:
                print("Test data download complete.")

    @staticmethod
    def __prepare_val_dataset(dataset, use_subset=False, subset_name="val_subset.json",
                              subset_size=250,
                              images_folder_default_name="val2017",
                              annotations_filename="person_keypoints_val2017.json",
                              verbose=True):
        """
        This internal method prepares the validation dataset depending on what type of dataset is provided.

        If an ExternalDataset object type is provided, the method tried to prepare the dataset based on the original
        implementation, supposing that the dataset is in the COCO format. The path provided is searched for the
        images folder and the annotations file, converts the annotations file into the internal format used if needed
        and finally the CocoValDataset object is returned.

        If the dataset is of the DatasetIterator format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.

        :param dataset: the dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param use_subset: whether to return a subset of the validation dataset, defaults to 'False'
        :type use_subset: bool, optional
        :param subset_name: the .json file where the validation dataset subset is saved, defaults to "val_subset.json"
        :type subset_name: str, optional
        :param subset_size: the size of the subset, defaults to 250
        :type subset_size: int
        :param images_folder_default_name: the name of the folder that contains the image files, defaults to "val2017"
        :type images_folder_default_name: str, optional
        :param annotations_filename: the .json file that contains the original annotations, defaults
            to "person_keypoints_val2017.json"
        :type annotations_filename: str, optional
        :param verbose: whether to print additional information, defaults to 'True'
        :type verbose: bool, optional

        :raises UserWarning: UserWarnings with appropriate messages are raised for wrong type of dataset, or wrong paths
            and filenames

        :return: returns CocoValDataset object or custom DatasetIterator implemented by user
        :rtype: CocoValDataset class object or DatasetIterator instance
        """
        if isinstance(dataset, ExternalDataset):
            if dataset.dataset_type.lower() != "coco":
                raise UserWarning("dataset_type must be \"COCO\"")

            # Get files and subdirectories of dataset.path directory
            f = []
            dirs = []
            for (dirpath, dirnames, filenames) in os.walk(dataset.path):
                f = filenames
                dirs = dirnames
                break

            # Get images folder
            if images_folder_default_name not in dirs:
                raise UserWarning("Didn't find \"" + images_folder_default_name +
                                  "\" folder in the dataset path provided.")
            images_folder = os.path.join(dataset.path, images_folder_default_name)

            # Get annotations file
            if annotations_filename not in f:
                raise UserWarning("Didn't find \"" + annotations_filename +
                                  "\" file in the dataset path provided.")
            val_labels_file = os.path.join(dataset.path, annotations_filename)

            if use_subset:
                val_sub_labels_file = os.path.join(dataset.path, subset_name)
                if subset_name not in f:
                    if verbose:
                        print("Didn't find " + subset_name + " in dataset.path, creating new...")
                    make_val_subset.make_val_subset(val_labels_file,
                                                    output_path=val_sub_labels_file,
                                                    num_images=subset_size)
                    if verbose:
                        print("Created new validation subset file.")
                    data = CocoValDataset(val_sub_labels_file, images_folder)
                else:
                    if verbose:
                        print("Val subset already exists.")
                    data = CocoValDataset(val_sub_labels_file, images_folder)
                    if len(data) != subset_size:
                        if verbose:
                            print("Val subset is wrong size, creating new.")
                        # os.remove(val_sub_labels_file)
                        make_val_subset.make_val_subset(val_labels_file,
                                                        output_path=val_sub_labels_file,
                                                        num_images=subset_size)
                        if verbose:
                            print("Created new validation subset file.")
                        data = CocoValDataset(val_sub_labels_file, images_folder)
            else:
                data = CocoValDataset(val_labels_file, images_folder)
            return data

        elif isinstance(dataset, DatasetIterator):
            return dataset

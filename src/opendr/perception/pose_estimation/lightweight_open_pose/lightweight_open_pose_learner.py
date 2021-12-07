# Copyright 2020-2021 OpenDR European Project
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
import onnxruntime as ort
import os
import ntpath
import shutil
import cv2
import torch
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import DataParallel
from tensorboardX import SummaryWriter
from torchvision import transforms
from urllib.request import urlretrieve

# OpenDR engine imports
from opendr.engine.learners import Learner
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from opendr.engine.data import Image
from opendr.engine.target import Pose
from opendr.engine.constants import OPENDR_SERVER_URL

# OpenDR lightweight_open_pose imports
from opendr.perception.pose_estimation.lightweight_open_pose.filtered_pose import FilteredPose
from opendr.perception.pose_estimation.lightweight_open_pose.utilities import track_poses
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.models.with_mobilenet import \
    PoseEstimationWithMobileNet
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.models.with_mobilenet_v2 import \
    PoseEstimationWithMobileNetV2
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.models.with_shufflenet import \
    PoseEstimationWithShuffleNet
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.get_parameters import \
    get_parameters_conv, get_parameters_bn, get_parameters_conv_depthwise
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.load_state import \
    load_state  # , load_from_mobilenet
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.loss import l2_loss
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.modules.keypoints import \
    extract_keypoints, group_keypoints
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.datasets.coco import CocoTrainDataset
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.datasets.coco import CocoValDataset
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.datasets.transformations import \
    ConvertKeypoints, Scale, Rotate, CropPad, Flip
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.val import \
    convert_to_coco_format, run_coco_eval, normalize, pad_width
from opendr.perception.pose_estimation.lightweight_open_pose.algorithm.scripts import \
    prepare_train_labels, make_val_subset

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader


class LightweightOpenPoseLearner(Learner):
    def __init__(self, lr=4e-5, epochs=280, batch_size=80, device='cuda', backbone='mobilenet',
                 lr_schedule='', temp_path='temp', checkpoint_after_iter=5000, checkpoint_load_iter=0,
                 val_after=5000, log_after=100, mobilenet_use_stride=True, mobilenetv2_width=1.0, shufflenet_groups=3,
                 num_refinement_stages=2, batches_per_iter=1,
                 experiment_name='default', num_workers=8, weights_only=True, output_name='detections.json',
                 multiscale=False, scales=None, visualize=False, base_height=256,
                 img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1 / 256), pad_value=(0, 0, 0),
                 half_precision=False):
        super(LightweightOpenPoseLearner, self).__init__(lr=lr, batch_size=batch_size, lr_schedule=lr_schedule,
                                                         checkpoint_after_iter=checkpoint_after_iter,
                                                         checkpoint_load_iter=checkpoint_load_iter,
                                                         temp_path=temp_path, device=device, backbone=backbone)
        self.parent_dir = temp_path  # Parent dir should be filled by the user according to README

        self.num_refinement_stages = num_refinement_stages  # How many extra refinement stages to add
        self.batches_per_iter = batches_per_iter
        self.epochs = epochs
        self.log_after = log_after
        self.val_after = val_after
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
        self.img_mean = img_mean
        self.img_scale = img_scale
        self.pad_value = pad_value
        self.previous_poses = []

        self.ort_session = None  # ONNX runtime inference session
        self.model_train_state = True

    def fit(self, dataset, val_dataset=None, logging_path='', logging_flush_secs=30,
            silent=False, verbose=True, epochs=None, use_val_subset=True, val_subset_size=250,
            images_folder_name="train2017", annotations_filename="person_keypoints_train2017.json",
            val_images_folder_name="val2017", val_annotations_filename="person_keypoints_val2017.json"):
        """
        This method is used for training the algorithm on a train dataset and validating on a val dataset.

        :param dataset: object that holds the training dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param val_dataset: object that holds the validation dataset, defaults to 'None'
        :type val_dataset: ExternalDataset class object or DatasetIterator class object, optional
        :param logging_path: path to save tensorboard log files. If set to None or '', tensorboard logging is
            disabled, defaults to ''
        :type logging_path: str, optional
        :param logging_flush_secs: how often, in seconds, to flush the tensorboard data to disk, defaults to '30'
        :type logging_flush_secs: int, optional
        :param silent: if set to True, disables all printing of training progress reports and other information
            to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :param epochs: overrides epochs attribute set in constructor, defaults to 'None'
        :type epochs: int, optional
        :param use_val_subset: if set to True, a subset of the validation dataset is created and used in
            evaluation, defaults to 'True'
        :type use_val_subset: bool, optional
        :param val_subset_size: controls the size of the validation subset, defaults to '250'
        :type val_subset_size: int, optional
        :param images_folder_name: folder name that contains the dataset images. This folder should be contained in
            the dataset path provided. Note that this is a folder name, not a path, defaults to 'train2017'
        :type images_folder_name: str, optional
        :param annotations_filename: filename of the annotations json file. This file should be contained in the
            dataset path provided, defaults to 'person_keypoints_train2017.json'
        :type annotations_filename: str, optional
        :param val_images_folder_name: folder name that contains the validation images. This folder should be contained
            in the dataset path provided. Note that this is a folder name, not a path, defaults to 'val2017'
        :type val_images_folder_name: str, optional
        :param val_annotations_filename: filename of the validation annotations json file. This file should be
            contained in the dataset path provided, defaults to 'person_keypoints_val2017.json'
        :type val_annotations_filename: str, optional

        :return: returns stats regarding the last evaluation ran
        :rtype: dict
        """
        # Training dataset initialization
        data = self.__prepare_dataset(dataset, stride=self.stride,
                                      prepared_annotations_name="prepared_train_annotations.pkl",
                                      images_folder_default_name=images_folder_name,
                                      annotations_filename=annotations_filename,
                                      verbose=verbose and not silent)
        train_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers)
        batches = int(len(data) / self.batch_size)

        # Tensorboard logging
        if logging_path != '' and logging_path is not None:
            logging = True
            file_writer = SummaryWriter(logging_path, flush_secs=logging_flush_secs)
        else:
            logging = False
            file_writer = None

        # Model initialization
        if self.model is None:
            self.init_model()

        checkpoints_folder = os.path.join(self.parent_dir, '{}_checkpoints'.format(self.experiment_name))
        if self.checkpoint_after_iter != 0 and not os.path.exists(checkpoints_folder):
            # User set checkpoint_after_iter so checkpoints need to be created
            # Checkpoints folder was just created
            os.makedirs(checkpoints_folder)

        checkpoint = None
        if self.checkpoint_load_iter == 0:
            # User set checkpoint_load_iter to 0, so they want to train from scratch
            self.download(mode="weights", verbose=verbose and not silent)
            backbone_weights_path = None
            if self.backbone == "mobilenet":
                backbone_weights_path = os.path.join(self.parent_dir, "mobilenet_sgd_68.848.pth.tar")
            elif self.backbone == "mobilenetv2":
                backbone_weights_path = os.path.join(self.parent_dir, "mobilenetv2_1.0-f2a8633.pth.tar")
            elif self.backbone == "shufflenet":
                backbone_weights_path = os.path.join(self.parent_dir, "shufflenet.pth.tar")
            try:
                checkpoint = torch.load(backbone_weights_path, map_location=torch.device(self.device))
            except FileNotFoundError as e:
                e.strerror = "Pretrained weights 'pth.tar' file must be placed in temp_path provided. \n " \
                             "No such file or directory."
                raise e
            if not silent and verbose:
                print("Loading default weights:", backbone_weights_path)
        else:
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

        if not silent and verbose:
            print("Model trainable parameters:", self.count_parameters())

        optimizer = optim.Adam([
            {'params': get_parameters_conv(self.model.model, 'weight')},
            {'params': get_parameters_conv_depthwise(self.model.model, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(self.model.model, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(self.model.model, 'bias'), 'lr': self.lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv(self.model.cpm, 'weight'), 'lr': self.lr},
            {'params': get_parameters_conv(self.model.cpm, 'bias'), 'lr': self.lr * 2, 'weight_decay': 0},
            {'params': get_parameters_conv_depthwise(self.model.cpm, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_conv(self.model.initial_stage, 'weight'), 'lr': self.lr},
            {'params': get_parameters_conv(self.model.initial_stage, 'bias'), 'lr': self.lr * 2,
             'weight_decay': 0},
            {'params': get_parameters_conv(self.model.refinement_stages, 'weight'), 'lr': self.lr * 4},
            {'params': get_parameters_conv(self.model.refinement_stages, 'bias'), 'lr': self.lr * 8,
             'weight_decay': 0},
            {'params': get_parameters_bn(self.model.refinement_stages, 'weight'), 'weight_decay': 0},
            {'params': get_parameters_bn(self.model.refinement_stages, 'bias'), 'lr': self.lr * 2,
             'weight_decay': 0},
        ], lr=self.lr, weight_decay=5e-4)

        num_iter = 0
        current_epoch = 0
        drop_after_epoch = [100, 200, 260]

        if self.lr_schedule != '':
            scheduler = self.lr_schedule(optimizer)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.333)

        if not self.weights_only and self.checkpoint_load_iter != 0:
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                if self.device == "cuda":
                    # Move optimizer state to cuda
                    # Taken from https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
                scheduler.load_state_dict(checkpoint['scheduler'])
                num_iter = checkpoint['iter']
                current_epoch = checkpoint['current_epoch']
            except ValueError as e:
                raise e
        elif self.checkpoint_load_iter != 0:
            num_iter = self.checkpoint_load_iter

        if self.device == "cuda":
            self.model = DataParallel(self.model)
        self.model.train()
        if self.device == "cuda":
            self.model = self.model.cuda()

        if epochs is not None:
            self.epochs = epochs
        eval_results = {}
        eval_results_list = []
        paf_losses = []
        heatmap_losses = []
        for epochId in range(current_epoch, self.epochs):
            total_losses = [0, 0] * (self.num_refinement_stages + 1)  # heatmaps loss, paf loss per stage
            batch_per_iter_idx = 0

            pbar = None
            pbarDesc = ""
            batch_index = 0
            if not silent:
                pbarDesc = "Epoch #" + str(epochId) + " progress"
                pbar = tqdm(desc=pbarDesc, total=batches, bar_format="{l_bar}%s{bar}{r_bar}" % '\x1b[38;5;231m')
            for batch_data in train_loader:
                if batch_per_iter_idx == 0:
                    optimizer.zero_grad()
                images = batch_data['image']
                keypoint_masks = batch_data['keypoint_mask']
                paf_masks = batch_data['paf_mask']
                keypoint_maps = batch_data['keypoint_maps']
                paf_maps = batch_data['paf_maps']
                if self.device == "cuda":
                    images = images.cuda()
                    keypoint_masks = keypoint_masks.cuda()
                    paf_masks = paf_masks.cuda()
                    keypoint_maps = keypoint_maps.cuda()
                    paf_maps = paf_maps.cuda()

                stages_output = self.model(images)
                losses = []
                for loss_idx in range(len(total_losses) // 2):
                    losses.append(l2_loss(stages_output[loss_idx * 2], keypoint_maps, keypoint_masks, images.shape[0]))
                    losses.append(l2_loss(stages_output[loss_idx * 2 + 1], paf_maps, paf_masks, images.shape[0]))
                    total_losses[loss_idx * 2] += losses[-2].item() / self.batches_per_iter
                    total_losses[loss_idx * 2 + 1] += losses[-1].item() / self.batches_per_iter

                loss = losses[0]
                for loss_idx in range(1, len(losses)):
                    loss += losses[loss_idx]
                loss /= self.batches_per_iter
                loss.backward()
                batch_per_iter_idx += 1
                if batch_per_iter_idx == self.batches_per_iter:
                    optimizer.step()
                    batch_per_iter_idx = 0
                    num_iter += 1
                else:
                    # This loop is skipped here so tqdm and batch_index need to be updated
                    if not silent:
                        pbar.update(1)
                    batch_index += 1
                    continue

                paf_losses.append([])
                heatmap_losses.append([])
                for loss_idx in range(len(total_losses) // 2):
                    paf_losses[-1].append(total_losses[loss_idx * 2 + 1])
                    heatmap_losses[-1].append(total_losses[loss_idx * 2])

                if self.log_after != 0 and num_iter % self.log_after == 0:
                    if logging:
                        for loss_idx in range(len(total_losses) // 2):
                            file_writer.add_scalar(tag="stage" + str(loss_idx + 1) + "_paf_loss",
                                                   scalar_value=total_losses[loss_idx * 2 + 1] / self.log_after,
                                                   global_step=num_iter)
                            file_writer.add_scalar(tag="stage" + str(loss_idx + 1) + "_heatmaps_loss",
                                                   scalar_value=total_losses[loss_idx * 2] / self.log_after,
                                                   global_step=num_iter)
                    if not silent and verbose:
                        print('Iter: {}'.format(num_iter))
                        for loss_idx in range(len(total_losses) // 2):
                            print('\n'.join(['stage{}_pafs_loss:     {}', 'stage{}_heatmaps_loss: {}']).format(
                                loss_idx + 1, total_losses[loss_idx * 2 + 1] / self.log_after,
                                loss_idx + 1, total_losses[loss_idx * 2] / self.log_after))
                    for loss_idx in range(len(total_losses)):
                        total_losses[loss_idx] = 0
                if self.checkpoint_after_iter != 0 and num_iter % self.checkpoint_after_iter == 0:
                    snapshot_name = '{}/checkpoint_iter_{}.pth'.format(checkpoints_folder, num_iter)
                    # Save checkpoint with full information for training state
                    self.__save(path=snapshot_name, optimizer=optimizer, scheduler=scheduler,
                                iter_=num_iter, current_epoch=epochId)

                if self.val_after != 0 and num_iter % self.val_after == 0 and val_dataset is not None:
                    if not silent and verbose:
                        print('Validation...')
                        eval_verbose = True
                    else:
                        eval_verbose = False
                    if not silent:
                        pbar.close()  # Close outer tqdm
                    eval_results = self.eval(val_dataset, silent=silent, verbose=eval_verbose,
                                             use_subset=use_val_subset, subset_size=val_subset_size,
                                             images_folder_name=val_images_folder_name,
                                             annotations_filename=val_annotations_filename)
                    eval_results_list.append(eval_results)
                    if not silent:
                        # Re-initialize outer tqdm
                        pbar = tqdm(desc=pbarDesc, initial=batch_index, total=batches,
                                    bar_format="{l_bar}%s{bar}{r_bar}" % '\x1b[38;5;231m')
                    if logging:
                        file_writer.add_scalar(tag="Average Precision @IoU=0.5:0.95, area = all",
                                               scalar_value=eval_results["average_precision"][0],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Precision @IoU=0.5, area = all",
                                               scalar_value=eval_results["average_precision"][1],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Precision @IoU=0.75, area = all",
                                               scalar_value=eval_results["average_precision"][2],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Precision @IoU=0.5:0.95, area = medium",
                                               scalar_value=eval_results["average_precision"][3],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Precision @IoU=0.5:0.95, area = large",
                                               scalar_value=eval_results["average_precision"][4],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Recall @IoU=0.5:0.95, area = all",
                                               scalar_value=eval_results["average_precision"][0],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Recall @IoU=0.5, area = all",
                                               scalar_value=eval_results["average_precision"][1],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Recall @IoU=0.75, area = all",
                                               scalar_value=eval_results["average_precision"][2],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Recall @IoU=0.5:0.95, area = medium",
                                               scalar_value=eval_results["average_precision"][3],
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Recall @IoU=0.5:0.95, area = large",
                                               scalar_value=eval_results["average_precision"][4],
                                               global_step=num_iter)
                        avg_precision = np.mean(eval_results["average_precision"])
                        file_writer.add_scalar(tag="Average Precision - all",
                                               scalar_value=avg_precision,
                                               global_step=num_iter)
                        avg_recall = np.mean(eval_results["average_recall"])
                        file_writer.add_scalar(tag="Average Recall - all",
                                               scalar_value=avg_recall,
                                               global_step=num_iter)
                        file_writer.add_scalar(tag="Average Score - all",
                                               scalar_value=np.mean([avg_precision, avg_recall]),
                                               global_step=num_iter)
                        file_writer.flush()  # manually flush eval results to disk
                if not silent:
                    pbar.update(1)
                batch_index += 1
            if not silent:
                pbar.close()
            scheduler.step()
        if logging:
            file_writer.close()
        # Return a dict of lists of PAF and Heatmap losses per stage and a list of all evaluation results dictionaries
        if self.half and self.device == 'cuda':
            self.model.half()

        return {"paf_losses": paf_losses, "heatmap_losses": heatmap_losses, "eval_results_list": eval_results_list}

    def eval(self, dataset, silent=False, verbose=True, use_subset=True, subset_size=250,
             images_folder_name="val2017", annotations_filename="person_keypoints_val2017.json"):
        """
        This method is used to evaluate a trained model on an evaluation dataset.

        :param dataset: object that holds the evaluation dataset.
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param silent: if set to True, disables all printing of evalutaion progress reports and other information
            to STDOUT, defaults to 'False'
        :type silent: bool, optional
        :param verbose: if set to True, enables the maximum verbosity, defaults to 'True'
        :type verbose: bool, optional
        :param use_subset: If set to True, a subset of the validation dataset is created and used in
            evaluation, defaults to 'True'
        :type use_subset: bool, optional
        :param subset_size: Controls the size of the validation subset, defaults to '250'
        :type subset_size: int, optional
        :param images_folder_name: Folder name that contains the dataset images. This folder should be contained in
            the dataset path provided. Note that this is a folder name, not a path, defaults to 'val2017'
        :type images_folder_name: str, optional
        :param annotations_filename: Filename of the annotations json file. This file should be contained in the
            dataset path provided, defaults to 'pesron_keypoints_val2017.json'
        :type annotations_filename: str, optional

        :returns: returns stats regarding evaluation
        :rtype: dict
        """
        # Validation dataset initialization
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
        if self.device == "cuda":
            self.model = self.model.cuda()
            if self.half:
                self.model.half()

        if self.multiscale:
            self.scales = [0.5, 1.0, 1.5, 2.0]

        coco_result = []

        pbar_eval = None
        if not silent:
            pbarDesc = "Evaluation progress"
            pbar_eval = tqdm(desc=pbarDesc, total=len(data), bar_format="{l_bar}%s{bar}{r_bar}" % '\x1b[38;5;231m')
        for sample in data:
            file_name = sample['file_name']
            img = sample['img']
            avg_heatmaps, avg_pafs, _, _ = self.__infer_eval(img)
            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(18):  # 19th for bg
                total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type,
                                                         total_keypoints_num)
            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, avg_pafs)
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
        if not silent:
            pbar_eval.close()
        if self.model_train_state:
            self.model = self.model.train()  # Revert model state to train

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

    def infer(self, img, upsample_ratio=4, track=True, smooth=True):
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
        if self.device == "cuda":
            tensor_img = tensor_img.cuda()
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

    def __save(self, path, optimizer, scheduler, iter_, current_epoch):
        """
        Internal save implementation is used to create checkpoints. Provided with a path,
        it adds training state information in a custom dict, optimizer and scheduler state_dicts,
        iteration number and current epoch id.

        :param path: path for the model to be saved
        :type path: str
        :param optimizer: the optimizer used for training
        :type optimizer: Optimizer PyTorch object
        :param scheduler: the scheduler used for training
        :type scheduler: Scheduler PyTorch object
        :param iter_: the current iteration number
        :type iter_: int
        :param current_epoch: the current epoch id
        :type current_epoch: int
        """
        custom_dict = {'state_dict': self.model.module.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'iter': iter_, 'current_epoch': current_epoch}
        torch.save(custom_dict, path)

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
        if self.device == "cuda":
            self.model.cuda()
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

    def __convert_to_onnx(self, output_name, do_constant_folding=False, verbose=False):
        """
        Converts the loaded regular PyTorch model to an ONNX model and saves it to disk.

        :param output_name: path and name to save the model, e.g. "/models/onnx_model.onnx"
        :type output_name: str
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        width = 344
        if self.device == "cuda":
            inp = torch.randn(1, 3, self.base_height, width).cuda()
        else:
            inp = torch.randn(1, 3, self.base_height, width)
        input_names = ['data']
        output_names = ['stage_0_output_1_heatmaps', 'stage_0_output_0_pafs',
                        'stage_1_output_1_heatmaps', 'stage_1_output_0_pafs']

        torch.onnx.export(self.model, inp, output_name, verbose=verbose, enable_onnx_checker=True,
                          do_constant_folding=do_constant_folding, input_names=input_names, output_names=output_names,
                          dynamic_axes={"data": {3: "width"}})

    def optimize(self, do_constant_folding=False):
        """
        Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.

        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        if self.model is None:
            raise UserWarning("No model is loaded, cannot optimize. Load or train a model first.")
        if self.ort_session is not None:
            raise UserWarning("Model is already optimized in ONNX.")

        try:
            self.__convert_to_onnx(os.path.join(self.temp_path, "onnx_model_temp.onnx"), do_constant_folding)
        except FileNotFoundError:
            # Create temp directory
            os.makedirs(self.temp_path, exist_ok=True)
            self.__convert_to_onnx(os.path.join(self.temp_path, "onnx_model_temp.onnx"), do_constant_folding)

        self.__load_from_onnx(os.path.join(self.temp_path, "onnx_model_temp.onnx"))

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError

    def count_parameters(self):
        """
        Returns the number of the model's trainable parameters.

        :return: number of trainable parameters
        :rtype: int
        """
        if self.model is None:
            raise UserWarning("Model is not initialized, can't count trainable parameters.")
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

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
            file_url = os.path.join(url, "dataset", "image", "000000000785.jpg")
            urlretrieve(file_url, os.path.join(self.temp_path, "dataset", "image", "000000000785.jpg"))

            if verbose:
                print("Test data download complete.")

    def __infer_eval(self, img):
        """
        Internal infer method used for evaluation. This infer can run on multiple scale ratios, depending on
        the self.scales attribute, performing multiple passes over the image averaging the results. This generally
        produces better results.

        :param img: Image to run infer on
        :type img: Image class object
        """
        if not isinstance(img, Image):
            img = Image(img)

        # Bring image into the appropriate format for the implementation
        img = img.convert(format='channels_last', channel_order='bgr')

        img_mean = self.img_mean  # Defaults to (128, 128, 128)
        img_scale = self.img_scale  # Defaults to 1 / 256
        pad_value = self.pad_value  # Defaults to (0, 0, 0)
        base_height = self.base_height  # Defaults to 256
        scales = self.scales  # Defaults to [1]
        stride = self.stride  # Defaults to 8

        normed_img = normalize(img, img_mean, img_scale)
        height, width, _ = normed_img.shape
        scales_ratios = [scale * base_height / float(height) for scale in scales]
        avg_heatmaps = np.zeros((height, width, 19), dtype=np.float32)
        avg_pafs = np.zeros((height, width, 38), dtype=np.float32)

        pad = None
        for ratio in scales_ratios:
            scaled_img = cv2.resize(normed_img, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
            min_dims = [base_height, max(scaled_img.shape[1], base_height)]
            padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
            if self.device == "cuda":
                tensor_img = tensor_img.cuda()
                if self.half:
                    tensor_img = tensor_img.half()
            stages_output = self.model(tensor_img)

            stage2_heatmaps = stages_output[-2]
            heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
            if self.half:
                heatmaps = np.float32(heatmaps)
            heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
            heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
            avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

            stage2_pafs = stages_output[-1]
            pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
            if self.half:
                pafs = np.float32(pafs)
            pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
            pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)
            avg_pafs = avg_pafs + pafs / len(scales_ratios)

        return avg_heatmaps, avg_pafs, scales_ratios, pad

    @staticmethod
    def __prepare_dataset(dataset, stride, prepared_annotations_name="prepared_train_annotations.pkl",
                          images_folder_default_name="train2017",
                          annotations_filename="person_keypoints_train2017.json",
                          verbose=True):
        """
        This internal method prepares the train dataset depending on what type of dataset is provided.

        If an ExternalDataset object type is provided, the method tried to prepare the dataset based on the original
        implementation, supposing that the dataset is in the COCO format. The path provided is searched for the
        images folder and the annotations file, converts the annotations file into the internal format used if needed
        and finally the CocoTrainDataset object is returned.

        If the dataset is of the DatasetIterator format, then it's a custom implementation of a dataset and all
        required operations should be handled by the user, so the dataset object is just returned.

        :param dataset: the dataset
        :type dataset: ExternalDataset class object or DatasetIterator class object
        :param stride: the stride attribute of the learner
        :type stride: int
        :param prepared_annotations_name: the .pkl file that should contain the converted annotations, defaults
            to "prepared_train_annotations.pkl"
        :type prepared_annotations_name: str, optional
        :param images_folder_default_name: the name of the folder that contains the image files, defaults to "train2017"
        :type images_folder_default_name: str, optional
        :param annotations_filename: the .json file that contains the original annotations, defaults
            to "person_keypoints_train2017.json"
        :type annotations_filename: str, optional
        :param verbose: whether to print additional information, defaults to 'True'
        :type verbose: bool, optional

        :raises UserWarning: UserWarnings with appropriate messages are raised for wrong type of dataset, or wrong paths
            and filenames

        :return: returns CocoTrainDataset object or custom DatasetIterator implemented by user
        :rtype: CocoTrainDataset class object or DatasetIterator instance
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
            annotations_file = os.path.join(dataset.path, annotations_filename)

            # Convert annotations to internal format if needed
            if prepared_annotations_name not in f:
                if verbose:
                    print("Didn't find " + prepared_annotations_name + " in dataset.path, creating new...")
                prepare_train_labels.convert_annotations(annotations_file,
                                                         output_path=os.path.join(dataset.path,
                                                                                  prepared_annotations_name))
                if verbose:
                    print("Created new .pkl file containing prepared annotations in internal format.")
            prepared_train_labels = os.path.join(dataset.path, prepared_annotations_name)

            sigma = 7
            paf_thickness = 1
            return CocoTrainDataset(prepared_train_labels, images_folder,
                                    stride, sigma, paf_thickness,
                                    transform=transforms.Compose([
                                        ConvertKeypoints(),
                                        Scale(),
                                        Rotate(pad=(128, 128, 128)),
                                        CropPad(pad=(128, 128, 128)),
                                        Flip()]))
        elif isinstance(dataset, DatasetIterator):
            return dataset

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

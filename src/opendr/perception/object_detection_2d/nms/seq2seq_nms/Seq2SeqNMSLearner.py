
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
import torch.nn.functional as F
import pickle
import numpy as np
import torchvision
import os
from opendr.engine.learners import Learner
from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.optim as optim
from tqdm import tqdm
import collections
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys
from opendr.engine.target import BoundingBox, BoundingBoxList
from opendr.engine.data import Image
from opendr.perception.object_detection_2d.nms.seq2seq_nms.algorithm.seq2seq_model import Seq2SeqNet
from opendr.perception.object_detection_2d.nms.seq2seq_nms.algorithm.fmod import FMoD
from opendr.perception.object_detection_2d.nms.seq2seq_nms.algorithm.dataset import Dataset_NMS


class Seq2SeqNMSLearner(Learner):
    def __init__(self, lr=0.0001, lr_schedule='', checkpoint_after_iter=1, checkpoint_load_iter=0,
                 experiment_name='default', temp_path='temp', device='cuda', use_fmod=True,
                 fmod_map_type='EDGEMAP_B', fmod_map_bin=True, dropout=0.02, fmod_roi_pooling_dim=160,
                 fmod_map_res_dim=800, fmod_pyramid_lvl=3, lq_dim=256, sq_dim=128, app_input_dim=None,
                 num_JPUs=4, pretrained_demo_model=None, log_after=500, iou_filtering=None, fmod_init_path=None):
        super(Seq2SeqNMSLearner, self).__init__(lr=lr, batch_size=1, lr_schedule=lr_schedule,
                                                checkpoint_after_iter=checkpoint_after_iter,
                                                checkpoint_load_iter=checkpoint_load_iter,
                                                temp_path=temp_path, device=device, backbone='default')
        self.use_fmod = use_fmod
        if self.use_fmod:
            self.fmod_normalization = None
            self.fmod_map_type = fmod_map_type
            self.fmod_roi_pooling_dim = [fmod_roi_pooling_dim, fmod_roi_pooling_dim]
            self.fmod_map_res_dim = fmod_map_res_dim
            self.fmod_pyramid_lvl = fmod_pyramid_lvl
            self.fmod_feats_dim = 0
            for i in range(0, self.fmod_pyramid_lvl):
                self.fmod_feats_dim = self.fmod_feats_dim + 15 * (pow(4, i))
            self.fmod_map_bin = fmod_map_bin
            self.app_input_dim = self.fmod_feats_dim
        else:
            if app_input_dim is None:
                raise Exception("The dimension of the input appearance-based features is not provided...")
            else:
                self.app_input_dim = app_input_dim
        self.geom_input_dim = 14
        self.lq_dim = lq_dim
        self.sq_dim = sq_dim
        self.dropout = dropout
        self.num_JPUs = num_JPUs
        self.parent_dir = temp_path
        if not os.path.isdir(self.parent_dir):
            os.mkdir(self.parent_dir)
        self.experiment_name = experiment_name
        if not os.path.isdir(os.path.join(self.parent_dir, self.experiment_name)):
            os.mkdir(os.path.join(self.parent_dir, self.experiment_name))
        self.pretrained_demo_model = pretrained_demo_model
        self.checkpoint_load_iter = checkpoint_load_iter
        self.log_after = log_after
        self.iou_filtering = iou_filtering
        self.classes = None
        self.class_ids = None
        self.device = device
        self.fMoD = None
        if self.use_fmod:
            self.fMoD = FMoD(roi_pooling_dim=self.fmod_roi_pooling_dim, pyramid_depth=self.fmod_pyramid_lvl,
                             resize_dim=self.fmod_map_res_dim,
                             map_type=self.fmod_map_type, map_bin=self.fmod_map_bin, device=self.device)
            if fmod_init_path is not None:
                fmod_mean_std = load_FMoD_init(fmod_init_path)
                self.fMoD.set_mean_std(mean_values=fmod_mean_std['mean'], std_values=fmod_mean_std['std'])

    def fit(self, dataset, val_dataset=None, epochs=None, logging_path='', logging_flush_secs=30, silent=True,
            verbose=True, nms_gt_iou=0.5, boxes_sorted=False, max_dt_boxes=400):

        datasets_folder = './datasets'
        dataset_nms = Dataset_NMS(datasets_folder, dataset, split='train')
        if self.classes is None:
            self.classes = dataset_nms.classes
            self.class_ids = dataset_nms.class_ids

        if logging_path != '' and logging_path is not None:
            logging = True
            file_writer = SummaryWriter(logging_path, flush_secs=logging_flush_secs)
        else:
            logging = False
            file_writer = None

        if self.model is None:
            self.init_model()
        checkpoints_folder = os.path.join(self.parent_dir, self.experiment_name, 'checkpoints')
        if self.checkpoint_after_iter != 0 and not os.path.exists(checkpoints_folder):
            os.makedirs(checkpoints_folder)

        if self.pretrained_demo_model is not None:
            self.download(mode="weights", verbose=verbose and not silent)
            weights_path = None
            if self.pretrained_demo_model == 'PETS':
                weights_path = os.path.join(self.parent_dir, "seq2seq_pets.pth.tar")
                self.checkpoint_load_iter = '?'
            elif self.pretrained_demo_model == 'COCO':
                weights_path = os.path.join(self.parent_dir, "seq2seq_coco.pth.tar")
                self.checkpoint_load_iter = '?'
            elif self.pretrained_demo_model == 'CrownHuman':
                weights_path = os.path.join(self.parent_dir, "seq2seq_crowdhuman.pth.tar")
                self.checkpoint_load_iter = '?'
            self.load(path=weights_path, verbose=verbose)
        elif self.checkpoint_load_iter != 0:
            checkpoint_name = "checkpoint_epoch_" + str(self.checkpoint_load_iter)
            checkpoint_full_path = os.path.join(checkpoints_folder, checkpoint_name)
            self.load(checkpoint_full_path)

        if not silent and verbose:
            print("Model trainable parameters:", self.count_parameters())

        self.model.train()
        if self.device == 'cuda':
            self.model = self.model.cuda()

        if epochs is None:
            raise ValueError("Training epochs not specified")
        elif epochs <= self.checkpoint_load_iter:
            raise ValueError("Training epochs are less than those of the loaded model")

        if self.use_fmod:
            fmod_mean_std = load_FMoD_init_from_dataset(dataset=dataset, map_type=self.fmod_map_type,
                                                        map_bin=self.fmod_map_bin, datasets_folder=datasets_folder)
            self.fMoD.set_mean_std(mean_values=fmod_mean_std['mean'], std_values=fmod_mean_std['std'])

        start_epoch = 0
        drop_after_epoch = [4, 7]

        train_ids = np.arange(len(dataset_nms.src_data))
        total_loss_iter = 0
        total_loss_epoch = 0
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=1e-9)  # HERE
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=drop_after_epoch, gamma=0.1)

        if self.checkpoint_load_iter != 0:
            checkpoint_name = "checkpoint_epoch_" + str(self.checkpoint_load_iter)
            checkpoint_full_path = os.path.join(checkpoints_folder, checkpoint_name)
            self.load(checkpoint_full_path)

        num_iter = 0
        training_weights = compute_class_weights(pos_weights=[0.5, 0.1], max_dets=max_dt_boxes, dataset_nms=dataset_nms)

        for epoch in range(start_epoch, epochs):
            pbar = None
            if not silent:
                pbarDesc = "Epoch #" + str(epoch) + " progress"
                pbar = tqdm(desc=pbarDesc, total=len(train_ids))
            np.random.shuffle(train_ids)
            for sample_id in train_ids:
                image_fln = dataset_nms.src_data[sample_id]['filename']
                loss = 0
                for class_index in range(len(dataset_nms.classes)):
                    if len(dataset_nms.src_data[sample_id]['dt_boxes'][class_index]) > 0:
                        dt_boxes = torch.tensor(
                            dataset_nms.src_data[sample_id]['dt_boxes'][class_index][:, 0:4]).float()
                        dt_scores = torch.tensor(dataset_nms.src_data[sample_id]['dt_boxes'][class_index][:, 4]).float()
                        if not boxes_sorted:
                            dt_scores, dt_scores_ids = torch.sort(dt_scores, descending=True)
                            dt_boxes = dt_boxes[dt_scores_ids]
                    else:
                        continue
                    gt_boxes = torch.tensor([]).float()
                    if len(dataset_nms.src_data[sample_id]['gt_boxes'][class_index]) > 0:
                        gt_boxes = torch.tensor(dataset_nms.src_data[sample_id]['gt_boxes'][class_index]).float()
                    image_path = os.path.join(datasets_folder, dataset, 'images', image_fln)
                    img_res = dataset_nms.src_data[sample_id]['resolution'][::-1]

                    if self.device == "cuda":
                        dt_boxes = dt_boxes.cuda()
                        dt_scores = dt_scores.cuda()
                        gt_boxes = gt_boxes.cuda()

                    val_ids = torch.logical_and((dt_boxes[:, 2] - dt_boxes[:, 0]) > 4,
                                                (dt_boxes[:, 3] - dt_boxes[:, 1]) > 4)
                    dt_boxes = dt_boxes[val_ids, :]
                    dt_scores = dt_scores[val_ids]

                    dt_boxes, dt_scores = drop_dets(dt_boxes, dt_scores)
                    if self.iou_filtering is not None and 1.0 > self.iou_filtering > 0:
                        dt_boxes, dt_scores = apply_torchNMS(boxes=dt_boxes, scores=dt_scores,
                                                             iou_thres=self.iou_filtering)

                    dt_boxes = dt_boxes[:max_dt_boxes]
                    dt_scores = dt_scores[:max_dt_boxes]
                    fmod_feats = None
                    if self.use_fmod:
                        img = Image.open(image_path)
                        img = img.convert(format='channels_last', channel_order='bgr')
                        self.fMoD.extract_maps(img=img, augm=True)
                        fmod_feats = self.fMoD.extract_FMoD_feats(dt_boxes)
                        fmod_feats = torch.unsqueeze(fmod_feats, dim=1)
                    msk = compute_mask(dt_boxes, iou_thres=0.2, extra=0.1)
                    q_geom_feats, k_geom_feats = compute_geometrical_feats(boxes=dt_boxes, scores=dt_scores,
                                                                           resolution=img_res)

                    optimizer.zero_grad()
                    preds = self.model(q_geom_feats=q_geom_feats, k_geom_feats=k_geom_feats, msk=msk,
                                       fmod_feats=fmod_feats)
                    preds = torch.clamp(preds, 0.001, 1 - 0.001)
                    labels = matching_module(scores=preds, dt_boxes=dt_boxes, gt_boxes=gt_boxes,
                                             iou_thres=nms_gt_iou)

                    # weights = (2.92 * labels + 0.932 * (1 - labels)).cuda()
                    weights = (training_weights[class_index][1] * labels + training_weights[class_index][0] * (
                            1 - labels)).cuda()

                    e = torch.distributions.uniform.Uniform(0.02, 0.0205).sample([labels.shape[0], 1])
                    if self.device == 'cuda':
                        e = e.cuda()
                    labels = labels * (1 - e) + (1 - labels) * e
                    ce_loss = F.binary_cross_entropy(preds, labels, reduction="none")
                    loss = loss + (ce_loss * weights).sum()
                    total_loss_iter = total_loss_iter + loss
                    total_loss_epoch = total_loss_epoch + loss
                loss.backward()
                optimizer.step()
                num_iter = num_iter + 1
                if self.log_after != 0 and num_iter % self.log_after == 0:
                    if logging:
                        file_writer.add_scalar(tag="cross entropy loss",
                                               scalar_value=total_loss_iter / self.log_after,
                                               global_step=num_iter)
                    if verbose:
                        print(''.join(['\nEpoch: {}',
                                       ' Iter: {}, cross entropy loss: {}']).format(epoch, num_iter,
                                                                                    total_loss_iter / self.log_after))
                    total_loss_iter = 0
                if not silent:
                    pbar.update(1)
            if not silent:
                pbar.close()
            if verbose:
                print(''.join(['Epoch: {}',
                               ' cross entropy loss: {}\n']).format(epoch,
                                                                    total_loss_epoch / len(train_ids)))
            if self.checkpoint_after_iter != 0 and epoch % self.checkpoint_after_iter == self.checkpoint_after_iter - 1:
                snapshot_name = '{}/checkpoint_epoch_{}'.format(checkpoints_folder, epoch)
                # Save checkpoint with full information for training state
                self.save(path=snapshot_name, optimizer=optimizer, scheduler=scheduler,
                          current_epoch=epoch, max_dt_boxes=max_dt_boxes)
            total_loss_epoch = 0
            scheduler.step()
        if logging:
            file_writer.close()
        # if not silent and verbose:
        #    print("Model trainable parameters:", self.count_parameters())

    def eval(self, dataset, verbose=True, split='test', boxes_sorted=False, max_dt_boxes=400, eval_folder=None):

        # Load dataset
        datasets_folder = './datasets'
        dataset_nms = Dataset_NMS(datasets_folder, dataset, split)

        if self.classes is None:
            self.classes = dataset_nms.classes
            self.class_ids = dataset_nms.class_ids

        annotations_filename = str.lower(dataset) + '_' + split + '.json'

        if eval_folder is None:
            eval_folder = os.path.join(self.parent_dir, self.experiment_name, 'eval')
        if not os.path.isdir(eval_folder):
            os.mkdir(eval_folder)
        output_file = os.path.join(eval_folder, 'detections.json')

        # Model initialization if needed
        if self.model is None and self.checkpoint_load_iter != 0:
            # No model loaded, initializing new
            self.init_model()
            checkpoint_name = "checkpoint_epoch_" + str(self.checkpoint_load_iter)
            checkpoint_folder = os.path.join(self.parent_dir, self.experiment_name, 'checkpoints')
            checkpoint_full_path = os.path.join(checkpoint_folder, checkpoint_name)
            self.load(path=checkpoint_full_path, verbose=verbose)

        elif self.model is None:
            raise AttributeError("self.model is None. Please load a model or set checkpoint_load_iter.")

        if self.use_fmod and (self.fMoD.mean is None or self.fMoD.std is None):
            fmod_mean_std = load_FMoD_init_from_dataset(dataset=dataset, map_type=self.fmod_map_type,
                                                        datasets_folder=datasets_folder)
            self.fMoD.set_mean_std(mean_values=fmod_mean_std['mean'], std_values=fmod_mean_std['std'])

        self.model = self.model.eval()  # Change model state to evaluation
        if self.device == "cuda":
            self.model = self.model.cuda()

        # Change model state to evaluation
        self.model = self.model.eval()
        if self.device == "cuda":
            self.model = self.model.cuda()

        train_ids = np.arange(len(dataset_nms.src_data))
        nms_results = []
        pbar_eval = None
        if verbose:
            pbarDesc = "Evaluation progress"
            pbar_eval = tqdm(desc=pbarDesc, total=len(train_ids))
        for sample_id in train_ids:
            image_fln = dataset_nms.src_data[sample_id]['filename']

            image_path = os.path.join(datasets_folder, dataset, 'images', image_fln)
            img_res = dataset_nms.src_data[sample_id]['resolution'][::-1]

            for class_index in range(len(dataset_nms.classes)):
                if len(dataset_nms.src_data[sample_id]['dt_boxes'][class_index]) > 0:
                    dt_boxes = torch.tensor(dataset_nms.src_data[sample_id]['dt_boxes'][class_index][:, 0:4]).float()
                    dt_scores = torch.tensor(dataset_nms.src_data[sample_id]['dt_boxes'][class_index][:, 4]).float()
                    if not boxes_sorted:
                        dt_scores, dt_scores_ids = torch.sort(dt_scores, descending=True)
                        dt_boxes = dt_boxes[dt_scores_ids]
                else:
                    continue

                if self.device == "cuda":
                    dt_boxes = dt_boxes.cuda()
                    dt_scores = dt_scores.cuda()

                val_ids = torch.logical_and((dt_boxes[:, 2] - dt_boxes[:, 0]) > 4,
                                            (dt_boxes[:, 3] - dt_boxes[:, 1]) > 4)
                dt_boxes = dt_boxes[val_ids, :]
                dt_scores = dt_scores[val_ids]

                if self.iou_filtering is not None and 1.0 > self.iou_filtering > 0:
                    dt_boxes, dt_scores = apply_torchNMS(boxes=dt_boxes, scores=dt_scores, iou_thres=self.iou_filtering)

                dt_boxes = dt_boxes[:max_dt_boxes]
                dt_scores = dt_scores[:max_dt_boxes]
                fmod_feats = None
                if self.use_fmod:
                    img = Image.open(image_path)
                    img = img.convert(format='channels_last', channel_order='bgr')
                    self.fMoD.extract_maps(img, augm=False)
                    fmod_feats = self.fMoD.extract_FMoD_feats(dt_boxes)
                    fmod_feats = torch.unsqueeze(fmod_feats, dim=1)
                msk = compute_mask(dt_boxes, iou_thres=0.2, extra=0.1)
                q_geom_feats, k_geom_feats = compute_geometrical_feats(boxes=dt_boxes, scores=dt_scores,
                                                                       resolution=img_res)
                with torch.no_grad():
                    preds = self.model(q_geom_feats=q_geom_feats, k_geom_feats=k_geom_feats, msk=msk,
                                       fmod_feats=fmod_feats).cpu().detach().numpy()
                    bboxes = dt_boxes.cpu().numpy().astype('float64')
                for j in range(len(preds)):
                    nms_results.append({
                        'image_id': dataset_nms.src_data[sample_id]['id'],
                        'bbox': [bboxes[j][0], bboxes[j][1], bboxes[j][2] - bboxes[j][0], bboxes[j][3] - bboxes[j][1]],
                        'category_id': dataset_nms.class_ids[dataset_nms.classes[class_index]],
                        'score': np.float64(preds[j])
                    })
            pbar_eval.update(1)
        pbar_eval.close()
        if verbose:
            print('Writing results json to {}'.format(output_file))
        with open(output_file, 'w') as fid:
            json.dump(nms_results, fid, indent=2)
        eval_result = run_coco_eval(gt_file_path=os.path.join(dataset_nms.path, 'annotations', annotations_filename),
                                    dt_file_path=output_file, only_classes=[1],
                                    verbose=verbose, max_dets=[max_dt_boxes])
        for i in range(len(eval_result)):
            print('Evaluation results (num_dets={})'.format(str(eval_result[i][1])))
            print(eval_result[i][0][0][1])
            print(eval_result[i][0][1][1])
            print(eval_result[i][0][2][1])
            print(eval_result[i][0][3][1])
            print('\n')

    def save(self, path, verbose=False, optimizer=None, scheduler=None, current_epoch=None, max_dt_boxes=800):
        """
        Method for saving the current model in the path provided
        :param path: path for the model to be saved
        :type path: str
        :param verbose: whether to print a success message or not, defaults to False
        :type verbose: bool, optional
        :param optimizer: the optimizer used for training
        :type optimizer: Optimizer PyTorch object
        :param scheduler: the scheduler used for training
        :type scheduler: Scheduler PyTorch object
        :param current_epoch: the current epoch id
        :type current_epoch: int
        """
        path = path.split('.')[0]
        custom_dict = {'state_dict': self.model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict(), 'current_epoch': current_epoch}
        torch.save(custom_dict, path + '.pth')

        metadata = {"model_paths": [os.path.basename(path) + '.pth'], "framework": "pytorch", "has_data": False,
                    "inference_params": {}, "optimized": False, "optimizer_info": {}, "backbone": {},
                    "format": "pth", "classes": self.classes, "use_fmod": self.use_fmod,
                    "lq_dim": self.lq_dim, "sq_dim": self.sq_dim, "num_JPUs": self.num_JPUs,
                    "geom_input_dim": self.geom_input_dim, "app_input_dim": self.app_input_dim,
                    "max_dt_boxes": max_dt_boxes}
        if self.use_fmod:
            metadata["fmod_map_type"] = self.fmod_map_type
            metadata["fmod_map_bin"] = self.fmod_map_bin
            metadata["fmod_roi_pooling_dim"] = self.fmod_roi_pooling_dim
            metadata["fmod_map_res_dim"] = self.fmod_map_res_dim
            metadata["fmod_pyramid_lvl"] = self.fmod_pyramid_lvl

        with open(path + '.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        if verbose:
            print("Saved Pytorch model.")

    def init_model(self):
        if self.model is None:
            self.model = Seq2SeqNet(dropout=self.dropout, use_fmod=self.use_fmod, app_input_dim=self.app_input_dim,
                                    geom_input_dim=self.geom_input_dim, lq_dim=self.lq_dim, sq_dim=self.sq_dim,
                                    num_JPUs=self.num_JPUs)
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            raise UserWarning("Tried to initialize model while model is already initialized.")

    def load(self, path, verbose=False):
        """
        Loads the model from inside the path provided, based on the metadata .json file included
        :param path: path of the checkpoint file was saved
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        """
        model_name = os.path.basename(os.path.normpath(path)).split('.')[0]
        dir_path = os.path.dirname(os.path.normpath(path))

        if verbose:
            print("Model name:", model_name, "-->", os.path.join(dir_path, model_name + ".json"))
        with open(os.path.join(dir_path, model_name + ".json")) as f:
            metadata = json.load(f)
        pth_path = os.path.join(dir_path, metadata["model_paths"][0])
        if verbose:
            print("Loading checkpoint:", pth_path)
        try:
            checkpoint = torch.load(pth_path, map_location=torch.device(self.device))
        except FileNotFoundError as e:
            e.strerror = "File " + pth_path + "not found."
            raise e

        self.assign_params(metadata=metadata, verbose=verbose)
        self.init_model()
        self.load_state(checkpoint)
        if self.device == 'cuda':
            self.model = self.model.cuda()
        if verbose:
            print("Loaded parameters and metadata.")
        return True

    def assign_params(self, metadata, verbose):

        if verbose and self.geom_input_dim is not None and self.geom_input_dim != metadata["geom_input_dim"]:
            print("Incompatible value for the attribute \"geom_input_dim\". It is now set to: " +
                  str(metadata["geom_input_dim"]))
        self.geom_input_dim = metadata["geom_input_dim"]
        if verbose and self.app_input_dim is not None and self.app_input_dim != metadata["app_input_dim"]:
            print("Incompatible value for the attribute \"app_input_dim\". It is now set to: " +
                  str(metadata["app_input_dim"]))
        self.app_input_dim = metadata["app_input_dim"]
        if verbose and self.use_fmod is not None and self.use_fmod != metadata["use_fmod"]:
            print("Incompatible value for the attribute \"use_fmod\". It is now set to: " +
                  str(metadata["use_fmod"]))
        self.use_fmod = metadata["use_fmod"]
        if verbose and self.fmod_map_type is not None and self.fmod_map_type != metadata["fmod_map_type"]:
            print("Incompatible value for the attribute \"fmod_map_type\". It is now set to: " +
                  str(metadata["fmod_map_type"]))
        self.fmod_map_type = metadata["fmod_map_type"]
        if verbose and self.fmod_map_bin is not None and self.fmod_map_bin != metadata["fmod_map_bin"]:
            print("Incompatible value for the attribute \"fmod_map_bin\". It is now set to: " +
                  str(metadata["fmod_map_bin"]))
        self.fmod_map_bin = metadata["fmod_map_bin"]
        if verbose and self.fmod_roi_pooling_dim is not None and \
                self.fmod_roi_pooling_dim != metadata["fmod_roi_pooling_dim"]:
            print("Incompatible value for the attribute \"fmod_roi_pooling_dim\". It is now set to: " +
                  str(metadata["fmod_roi_pooling_dim"]))
        self.fmod_roi_pooling_dim = metadata["fmod_roi_pooling_dim"]
        if verbose and self.fmod_map_res_dim is not None and \
                self.fmod_map_res_dim != metadata["fmod_map_res_dim"]:
            print("Incompatible value for the attribute \"fmod_map_res_dim\". It is now set to: " +
                  str(metadata["fmod_map_res_dim"]))
        self.fmod_map_res_dim = metadata["fmod_map_res_dim"]
        if verbose and self.fmod_pyramid_lvl is not None and \
                self.fmod_pyramid_lvl != metadata["fmod_pyramid_lvl"]:
            print("Incompatible value for the attribute \"fmod_pyramid_lvl\". It is now set to: " +
                  str(metadata["fmod_pyramid_lvl"]))
        self.fmod_pyramid_lvl = metadata["fmod_pyramid_lvl"]
        if verbose and self.lq_dim is not None and \
                self.lq_dim != metadata["lq_dim"]:
            print("Incompatible value for the attribute \"lq_dim\". It is now set to: " +
                  str(metadata["lq_dim"]))
        self.lq_dim = metadata["lq_dim"]
        if verbose and self.sq_dim is not None and self.sq_dim != metadata["sq_dim"]:
            print("Incompatible value for the attribute \"sq_dim\". It is now set to: " +
                  str(metadata["sq_dim"]))
        self.sq_dim = metadata["sq_dim"]
        if verbose and self.num_JPUs is not None and self.num_JPUs != metadata["num_JPUs"]:
            print("Incompatible value for the attribute \"num_JPUs\". It is now set to: " +
                  str(metadata["num_JPUs"]))
        self.num_JPUs = metadata["num_JPUs"]
        if verbose and 'max_dt_boxes' in metadata:
            print('Model is trained with as ' + str(metadata['max_dt_boxes']) + 'its maximum number of detections.')

    def load_state(self, checkpoint=None):
        if checkpoint is None:
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            try:
                source_state = checkpoint['state_dict']
            except KeyError:
                source_state = checkpoint
            target_state = self.model.state_dict()
            new_target_state = collections.OrderedDict()
            for target_key, target_value in target_state.items():
                if target_key in source_state and source_state[target_key].size() == target_state[target_key].size():
                    new_target_state[target_key] = source_state[target_key]
                else:
                    new_target_state[target_key] = target_state[target_key]
                    # print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))

            self.model.load_state_dict(new_target_state)

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
        print('ToDo')

    def infer(self, classes=None, dets=None, boxes_sorted=False, max_dt_boxes=1200, img_res=None, threshold=0.1):

        for class_index in range(len(classes)):
            if len(dets[class_index]) > 0:
                dt_boxes = dets[class_index][:, 0:4]
                dt_scores = dets[class_index][:, 4]
                if not boxes_sorted:
                    dt_scores, dt_scores_ids = torch.sort(dt_scores, descending=True)
                    dt_boxes = dt_boxes[dt_scores_ids]
            else:
                continue

            if self.device == "cuda":
                dt_boxes = dt_boxes.cuda()
                dt_scores = dt_scores.cuda()

            val_ids = torch.logical_and((dt_boxes[:, 2] - dt_boxes[:, 0]) > 4,
                                        (dt_boxes[:, 3] - dt_boxes[:, 1]) > 4)
            dt_boxes = dt_boxes[val_ids, :]
            dt_scores = dt_scores[val_ids]

            if self.iou_filtering is not None and 1.0 > self.iou_filtering > 0:
                dt_boxes, dt_scores = apply_torchNMS(boxes=dt_boxes, scores=dt_scores, iou_thres=self.iou_filtering)

            dt_boxes = dt_boxes[:max_dt_boxes]
            dt_scores = dt_scores[:max_dt_boxes]
            fmod_feats = None
            if self.use_fmod:
                fmod_feats = self.fMoD.extract_FMoD_feats(dt_boxes)
                fmod_feats = torch.unsqueeze(fmod_feats, dim=1)
            msk = compute_mask(dt_boxes, iou_thres=0.2, extra=0.1)
            q_geom_feats, k_geom_feats = compute_geometrical_feats(boxes=dt_boxes, scores=dt_scores,
                                                                   resolution=img_res)
            with torch.no_grad():
                preds = self.model(q_geom_feats=q_geom_feats, k_geom_feats=k_geom_feats, msk=msk,
                                   fmod_feats=fmod_feats).cpu().detach().numpy()
                bboxes = dt_boxes.cpu().numpy().astype('float64')

                mask = np.where(preds > threshold)[0]
                if mask.size == 0:
                    return BoundingBoxList([])
                preds = preds[mask]
                bboxes = bboxes[mask, :]

                bounding_boxes = BoundingBoxList([])
                for idx, box in enumerate(bboxes):
                    bbox = BoundingBox(left=box[0], top=box[1],
                                       width=box[2] - box[0],
                                       height=box[3] - box[1],
                                       name=class_index,
                                       score=preds[idx])
                    bounding_boxes.data.append(bbox)
        return bounding_boxes

    def optimize(self, **kwargs):
        """This method is not used in this implementation."""
        raise NotImplementedError

    def reset(self):
        """This method is not used in this implementation."""
        return NotImplementedError


def apply_torchNMS(boxes, scores, iou_thres):
    ids_nms = torchvision.ops.nms(boxes, scores, iou_thres)
    scores = scores[ids_nms]
    boxes = boxes[ids_nms]
    return boxes, scores


def compute_mask(boxes=None, iou_thres=0.2, extra=0.1):
    relations = filter_iou_boxes(boxes, iou_thres=iou_thres)
    mask1 = torch.tril(relations).float()
    mask2 = extra * torch.triu(relations, diagonal=1).float()
    mask = mask1 + mask2
    return mask


def filter_iou_boxes(boxes=None, iou_thres=0.2):
    ious = bb_intersection_over_union(boxes.unsqueeze(1).repeat(1, boxes.shape[0], 1),
                                      boxes.clone().unsqueeze(0).repeat(boxes.shape[0], 1, 1))
    ids_boxes = ious >= iou_thres
    return ids_boxes


def bb_intersection_over_union(boxAs=None, boxBs=None):
    xA = torch.maximum(boxAs[:, :, 0], boxBs[:, :, 0])
    yA = torch.maximum(boxAs[:, :, 1], boxBs[:, :, 1])
    xB = torch.minimum(boxAs[:, :, 2], boxBs[:, :, 2])
    yB = torch.minimum(boxAs[:, :, 3], boxBs[:, :, 3])
    interAreas = torch.maximum(torch.zeros_like(xB), xB - xA + 1) * torch.maximum(torch.zeros_like(yB), yB - yA + 1)
    boxAAreas = (boxAs[:, :, 2] - boxAs[:, :, 0] + 1) * (boxAs[:, :, 3] - boxAs[:, :, 1] + 1)
    boxBAreas = (boxBs[:, :, 2] - boxBs[:, :, 0] + 1) * (boxBs[:, :, 3] - boxBs[:, :, 1] + 1)
    ious = interAreas / (boxAAreas + boxBAreas - interAreas)
    return ious


def compute_geometrical_feats(boxes, scores, resolution):
    boxBs = boxes.clone().unsqueeze(0).repeat(boxes.shape[0], 1, 1)
    boxAs = boxes.unsqueeze(1).repeat(1, boxes.shape[0], 1)
    scoresBs = scores.unsqueeze(0).unsqueeze(-1).repeat(scores.shape[0], 1, 1)
    scoresAs = scores.unsqueeze(1).unsqueeze(1).repeat(1, scores.shape[0], 1)

    scale_div = [resolution[0] / 20, resolution[1] / 20]
    dx = ((boxBs[:, :, 0] - boxAs[:, :, 0] + boxBs[:, :, 2] - boxAs[:, :, 2]) / 2).unsqueeze(-1)
    dy = ((boxBs[:, :, 1] - boxAs[:, :, 1] + boxBs[:, :, 3] - boxAs[:, :, 3]) / 2).unsqueeze(-1)
    dxy = dx * dx + dy * dy
    dxy = dxy / (scale_div[0] * scale_div[0] + scale_div[1] * scale_div[1])
    dx = (dx / scale_div[0])
    dy = (dy / scale_div[1])
    sx = boxBs[:, :, 2] - boxBs[:, :, 0]
    sx_1 = (sx / (boxAs[:, :, 2] - boxAs[:, :, 0])).unsqueeze(-1)
    sx_2 = (sx / scale_div[0]).unsqueeze(-1)
    sy = boxBs[:, :, 3] - boxBs[:, :, 1]
    sy_1 = (sy / (boxAs[:, :, 3] - boxAs[:, :, 1])).unsqueeze(-1)
    sy_2 = (sy / scale_div[1]).unsqueeze(-1)
    scl = (boxBs[:, :, 2] - boxBs[:, :, 0]) * (boxBs[:, :, 3] - boxBs[:, :, 1])
    scl_1 = (scl / ((boxAs[:, :, 2] - boxAs[:, :, 0]) * (boxAs[:, :, 3] - boxAs[:, :, 1]))).unsqueeze(-1)
    scl_2 = (scl / (scale_div[0] * scale_div[1])).unsqueeze(-1)
    del scl

    scr_1 = 5 * scoresBs
    scr_2 = scr_1 - 5 * scoresAs

    sr_1 = torch.unsqueeze((boxBs[:, :, 3] - boxBs[:, :, 1]) / (boxBs[:, :, 2] - boxBs[:, :, 0]), dim=-1)
    sr_2 = torch.unsqueeze(((boxBs[:, :, 3] - boxBs[:, :, 1]) / (boxBs[:, :, 2] - boxBs[:, :, 0])) / (
            (boxAs[:, :, 3] - boxAs[:, :, 1]) / (boxAs[:, :, 2] - boxAs[:, :, 0])), dim=-1)

    ious = 5 * (bb_intersection_over_union(boxes.unsqueeze(1).repeat(1, boxes.shape[0], 1),
                                           boxes.clone().unsqueeze(0).repeat(boxes.shape[0], 1, 1))).unsqueeze(-1)
    enc_vers_all = torch.cat((dx, dy, dxy, sx_1, sx_2, sy_1, sy_2, ious, scl_1, scl_2, scr_1, scr_2, sr_1, sr_2), dim=2)
    enc_vers = enc_vers_all.diagonal(dim1=0, dim2=1).transpose(0, 1).unsqueeze(1)
    return enc_vers, enc_vers_all


def matching_module(scores, dt_boxes, gt_boxes, iou_thres, device='cuda'):
    sorted_indices = torch.argsort(-scores, dim=0)
    labels = torch.zeros(len(dt_boxes))
    assigned_GT = -torch.ones(len(gt_boxes))
    r = torch.tensor([-1, -1, -1, -1]).float().unsqueeze(0).unsqueeze(0)
    if device == 'cuda':
        r = r.cuda()
        labels = labels.cuda()
    for s in sorted_indices:
        gt_boxes_c = gt_boxes.clone().unsqueeze(0)
        gt_boxes_c[0, assigned_GT > -1, :] = r
        ious = bb_intersection_over_union(boxAs=dt_boxes[s].clone().unsqueeze(0), boxBs=gt_boxes_c)
        annot_iou, annot_box_id = torch.sort(ious.squeeze(), descending=True)
        if annot_box_id.ndim > 0:
            annot_box_id = annot_box_id[0]
            annot_iou = annot_iou[0]
        if annot_iou > iou_thres:
            assigned_GT[annot_box_id] = s
            labels[s] = 1
    return labels.unsqueeze(-1)


def run_coco_eval(dt_file_path=None, gt_file_path=None, only_classes=None, max_dets=None,
                  verbose=False):
    if max_dets is None:
        max_dets = [200, 400, 600, 800, 1000, 1200]
    results = []
    sys.stdout = open(os.devnull, 'w')
    for i in range(len(max_dets)):
        coco = COCO(gt_file_path)
        coco_dt = coco.loadRes(dt_file_path)
        cocoEval = COCOeval(coco, coco_dt, 'bbox')
        cocoEval.params.iouType = 'bbox'
        cocoEval.params.useCats = True
        cocoEval.params.catIds = only_classes
        cocoEval.params.maxDets = [max_dets[i]]
        cocoEval.evaluate()
        results.append([summarize_nms(coco_eval=cocoEval, maxDets=max_dets[i]), max_dets[i]])
        # print(results[i])
    del cocoEval, coco_dt, coco
    sys.stdout = sys.__stdout__
    return results


def summarize_nms(coco_eval=None, maxDets=100):
    def summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = coco_eval.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = coco_eval.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = coco_eval.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, aind, mind]
        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])
        stat_str = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return [mean_s, stat_str]

    def summarizeDets():
        stats = []
        stat, stat_str = summarize(1, maxDets=maxDets)
        stats.append([stat, stat_str])
        stat, stat_str = summarize(1, iouThr=.5, maxDets=maxDets)
        stats.append([stat, stat_str])
        stat, stat_str = summarize(1, iouThr=.75, maxDets=maxDets)
        stats.append([stat, stat_str])
        stat, stat_str = summarize(0, maxDets=maxDets)
        stats.append([stat, stat_str])
        return stats

    coco_eval.accumulate()
    summarized = summarizeDets()
    return summarized


def drop_dets(boxes, scores, keep_ratio=0.85):
    ids = np.arange(len(boxes))
    np.random.shuffle(ids)
    ids_keep = ids[0:int(len(boxes) * keep_ratio)]
    boxes_new = boxes[ids_keep, :]
    scores_new = scores[ids_keep]
    return boxes_new, scores_new


def load_FMoD_init_from_dataset(dataset=None, map_type='edgemap', datasets_folder='./datasets', map_bin=True):
    fmod_dir = os.path.join(datasets_folder, dataset, 'FMoD')
    if not os.path.exists(fmod_dir):
        os.makedirs(fmod_dir, exist_ok=True)
    map_type_c = map_type
    if map_bin:
        map_type_c = map_type_c + '_B'
    fmod_filename = dataset + '_' + map_type_c + '.pkl'
    fmod_filename = fmod_filename.lower()
    if not os.path.exists(os.path.join(fmod_dir, fmod_filename)):
        file_url = os.path.join(OPENDR_SERVER_URL + 'perception/non-maximum_suppression/FMoD', fmod_filename)
        try:
            urlretrieve(file_url, os.path.join(fmod_dir, fmod_filename))
        except Exception as e:
            raise e
    fmod_stats = load_FMoD_init(os.path.join(fmod_dir, fmod_filename))
    return fmod_stats


def load_FMoD_init(path=None):
    try:
        with open(path, 'rb') as fp:
            fmod_stats = pickle.load(fp)
            map_type = list(fmod_stats.keys())[0]
            fmod_stats = fmod_stats[map_type]
    except EnvironmentError as e:
        e.strerror = 'FMoD initialization .pkl file not found'
        raise e
    return fmod_stats


def compute_class_weights(pos_weights, max_dets=400, dataset_nms=None):
    num_pos = np.ones([len(dataset_nms.classes), 1])
    num_bg = np.ones([len(dataset_nms.classes), 1])
    weights = np.zeros([len(dataset_nms.classes), 2])
    for i in range(len(dataset_nms.src_data)):
        for cls_index in range(len(dataset_nms.classes)):
            num_pos[cls_index] = num_pos[cls_index] + \
                                   min(max_dets, len(dataset_nms.src_data[i]['gt_boxes'][cls_index]))
            num_bg[cls_index] = num_bg[cls_index] + max(0, min(max_dets,
                                                               len(dataset_nms.src_data[i]['dt_boxes'][cls_index])) -
                                                        min(max_dets, len(dataset_nms.src_data[i]['gt_boxes'][cls_index])))
    for class_index in range(len(dataset_nms.classes)):
        weights[class_index, 0] = (1 - pos_weights[class_index]) * (num_pos[class_index] +
                                                                    num_bg[class_index]) / num_bg[class_index]
        weights[class_index, 1] = pos_weights[class_index] * (num_pos[class_index] +
                                                              num_bg[class_index]) / num_pos[class_index]
    return weights

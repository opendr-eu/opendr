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

import os
import json
import torch
import ntpath
import shutil
import numpy as np
import onnxruntime as ort
from torchvision.transforms import transforms as T
from opendr.engine.learners import Learner
from opendr.engine.datasets import DatasetIterator, ExternalDataset, MappedDatasetIterator
from opendr.perception.object_tracking_2d.logger import Logger
from opendr.perception.object_tracking_2d.datasets.mot_dataset import JointDataset, RawMotDatasetIterator
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.models.model import create_model
from opendr.perception.object_tracking_2d.fair_mot.algorithm.run import train, evaluate
from opendr.perception.object_tracking_2d.fair_mot.algorithm.load import load_from_checkpoint
from opendr.perception.object_tracking_2d.datasets.mot_dataset import letterbox, process as process_dataset
from opendr.perception.object_tracking_2d.fair_mot.algorithm.lib.tracker.multitracker import JDETracker
from opendr.engine.data import Image
from opendr.engine.target import TrackingAnnotation, TrackingAnnotationList
from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve


class ObjectTracking2DFairMotLearner(Learner):
    def __init__(
        self,
        lr=0.0001,
        iters=-1,
        batch_size=4,
        optimizer="adam",
        lr_schedule="",
        backbone="dla_34",
        network_head="",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda",
        threshold=0.3,
        scale=1.0,
        lr_step=[20],
        head_conv=256,
        ltrb=True,
        num_classes=1,
        reg_offset=True,
        gpus=[0],
        num_workers=4,
        mse_loss=False,
        reg_loss='l1',
        dense_wh=False,
        cat_spec_wh=False,
        reid_dim=128,
        norm_wh=False,
        wh_weight=0.1,
        off_weight=1,
        id_weight=1,
        num_epochs=30,
        hm_weight=1,
        down_ratio=4,
        max_objs=500,
        track_buffer=30,
        image_mean=[0.408, 0.447, 0.47],
        image_std=[0.289, 0.274, 0.278],
        frame_rate=30,
        min_box_area=100,
    ):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(ObjectTracking2DFairMotLearner, self).__init__(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            backbone=backbone,
            network_head=network_head,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            temp_path=temp_path,
            device=device,
            threshold=threshold,
            scale=scale,
        )

        self.ltrb = ltrb
        self.head_conv = head_conv
        self.num_classes = num_classes
        self.reid_dim = reid_dim
        self.reg_offset = reg_offset
        self.gpus = gpus
        self.num_workers = num_workers
        self.mse_loss = mse_loss
        self.reg_loss = reg_loss
        self.dense_wh = dense_wh
        self.cat_spec_wh = cat_spec_wh
        self.reid_dim = reid_dim
        self.norm_wh = norm_wh
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.id_weight = id_weight
        self.num_epochs = num_epochs
        self.lr_step = lr_step
        self.hm_weight = hm_weight
        self.down_ratio = down_ratio
        self.max_objs = max_objs
        self.track_buffer = track_buffer
        self.image_mean = image_mean
        self.image_mean = image_mean
        self.image_std = image_std
        self.frame_rate = frame_rate
        self.min_box_area = min_box_area

        main_batch_size = self.batch_size // len(self.gpus)
        rest_batch_size = (self.batch_size - main_batch_size)
        self.chunk_sizes = [main_batch_size]

        for i in range(len(self.gpus) - 1):
            worker_chunk_size = rest_batch_size // (len(self.gpus) - 1)
            if i < rest_batch_size % (len(self.gpus) - 1):
                worker_chunk_size += 1
            self.chunk_sizes.append(worker_chunk_size)

        self.__create_model()

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
        path_no_folder_name = ''.join(path.rsplit(folder_name, 1))
        # If tail is '', then path was a/b/c/, which leaves a trailing double '/'
        if tail == '':
            path_no_folder_name = path_no_folder_name[0:-1]  # Remove one '/'

        # Create model directory
        new_path = path_no_folder_name + folder_name_no_ext
        os.makedirs(new_path, exist_ok=True)

        model_metadata = {"model_paths": [], "framework": "pytorch", "format": "", "has_data": False,
                          "inference_params": {}, "optimized": None, "optimizer_info": {}}

        if self.model.ort_session is None:
            model_metadata["model_paths"] = [
                folder_name_no_ext + ".pth",
            ]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pth"

            torch.save({
                'state_dict': self.model.state_dict()
            }, os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][0]))
            if verbose:
                print("Saved Pytorch model.")
        else:
            model_metadata["model_paths"] = [
                folder_name_no_ext + ".onnx"
            ]
            model_metadata["optimized"] = True
            model_metadata["format"] = "onnx"

            shutil.copy2(
                os.path.join(self.temp_path, "onnx_model_temp.onnx"),
                os.path.join(path_no_folder_name, folder_name_no_ext, model_metadata["model_paths"][0])
            )
            if verbose:
                print("Saved ONNX model.")

        with open(os.path.join(new_path, folder_name_no_ext + ".json"), 'w') as outfile:
            json.dump(model_metadata, outfile)

    def load(
        self,
        path,
        verbose=False,
    ):
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

        if not metadata["optimized"]:
            self.__load_from_pth(self.model, os.path.join(path, metadata["model_paths"][0]))
            if verbose:
                print("Loaded Pytorch model.")
        else:
            self.__load_rpn_from_onnx(os.path.join(path, metadata["model_paths"][0]))
            if verbose:
                print("Loaded ONNX model.")

    def reset(self):
        self.tracker.reset()

    def fit(
        self,
        dataset,
        val_dataset=None,
        val_epochs=-1,
        logging_path=None,
        silent=False,
        verbose=False,
        train_split_paths=None,
        val_split_paths=None,
        resume_optimizer=False,
        nID=None
    ):

        if train_split_paths is None:
            train_split_paths = {
                "mot20": os.path.join(
                    "perception", "object_tracking_2d", "datasets", "splits", "mot20.train"
                )
            }

        if val_split_paths is None:
            val_split_paths = train_split_paths

        logger = Logger(silent, verbose, logging_path)

        (
            input_dataset_iterator,
            eval_dataset_iterator,
        ) = self._prepare_datasets(
            dataset,
            val_dataset,
            train_split_paths,
            val_split_paths,
            require_val_dataset=val_epochs > 0,
        )

        if nID is None:
            nID = input_dataset_iterator.nID if hasattr(input_dataset_iterator, "nID") else dataset.nID

        checkpoints_path = os.path.join(self.temp_path, "checkpoints")
        if self.checkpoint_after_iter != 0 or self.checkpoint_load_iter != 0:
            os.makedirs(checkpoints_path, exist_ok=True)

        start_epoch = 0

        if self.checkpoint_load_iter != 0:
            _, _, start_epoch = load_from_checkpoint(
                self.model, os.path.join(checkpoints_path, f"checkpoint_{self.checkpoint_load_iter}.pth"),
                self.model_optimizer, resume_optimizer, self.lr, self.lr_step, log=logger.log,
            )

        last_eval_result = train(
            self.model,
            self.infer,
            self.model_optimizer,
            input_dataset_iterator,
            eval_dataset_iterator,
            self.batch_size,
            self.num_workers,
            self.gpus,
            self.chunk_sizes,
            self.iters,
            "train",  # exp_id,
            self.device,
            silent,  # hide_data_time,
            1 if verbose else (-1 if silent else 10),  # print_iter,
            self.mse_loss,
            self.reg_loss,
            self.dense_wh,
            self.cat_spec_wh,
            self.reid_dim,
            nID,
            self.norm_wh,
            1,  # num_stack,
            self.wh_weight,
            self.off_weight,
            self.id_weight,
            self.num_epochs,
            self.lr_step,
            self.temp_path,
            self.lr,
            self.reg_offset,
            self.hm_weight,
            checkpoints_path,
            self.checkpoint_after_iter,
            start_epoch,
            val_epochs=val_epochs,
            log=logger.log,
        )

        logger.close()

        return last_eval_result

    def eval(
        self,
        dataset,
        val_split_paths=None,
        logging_path=None,
        silent=False,
        verbose=False,
    ):

        logger = Logger(silent, verbose, logging_path)

        (
            _,
            eval_dataset_iterator,
        ) = self._prepare_datasets(
            None,
            dataset,
            None,
            val_split_paths,
            require_dataset=False,
        )

        result = evaluate(self.infer, dataset)

        logger.log(Logger.LOG_WHEN_NORMAL, result)

        logger.close()

        return result

    def infer(self, batch, frame_ids=None, img_size=(1088, 608)):

        if self.model is None:
            raise ValueError("No model loaded or created")

        self.model.eval()

        is_single_image = False

        if isinstance(batch, Image):
            batch = [batch]
            is_single_image = True
        elif not isinstance(batch, list):
            raise ValueError("Input batch should be an engine.Image or a list of engine.Image")

        if frame_ids is None:
            frame_ids = [-1] * len(batch)
        elif is_single_image:
            frame_ids = [frame_ids]

        results = []

        for image, frame_id in zip(batch, frame_ids):

            img0 = image.convert("channels_last", "bgr")  # BGR
            img, _, _, _ = letterbox(img0, height=img_size[1], width=img_size[0])

            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            blob = torch.from_numpy(img).to(self.device).unsqueeze(0)

            online_targets = self.tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

            result = TrackingAnnotationList([
                TrackingAnnotation(
                    name=0,
                    top=tlwh[0],
                    left=tlwh[1],
                    width=tlwh[2],
                    height=tlwh[3],
                    id=id,
                    score=score,
                    frame=frame_id,
                ) for tlwh, id, score in zip(
                    online_tlwhs,
                    online_ids,
                    online_scores
                )
            ])

            results.append(result)

        if is_single_image:
            results = results[0]

        return results

    def optimize(self, do_constant_folding=False, img_size=(1088, 608), optimizable_dcn_v2=False):
        """
        Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """

        if not optimizable_dcn_v2:
            raise Exception("Can not optimize the model while DCNv2 implementation is not optimizable")

        if self.model is None:
            raise UserWarning("No model is loaded, cannot optimize. Load or train a model first.")
        if self.model.ort_session is not None:
            raise UserWarning("Model is already optimized in ONNX.")

        input_shape = [
            1,
            3,
            img_size[1],
            img_size[0],
        ]

        try:
            self.__convert_to_onnx(
                input_shape,
                os.path.join(self.temp_path, "onnx_model_temp.onnx"), do_constant_folding
            )
        except FileNotFoundError:
            # Create temp directory
            os.makedirs(self.temp_path, exist_ok=True)
            self.__convert_rpn_to_onnx(
                input_shape,
                os.path.join(self.temp_path, "onnx_model_temp.onnx"), do_constant_folding
            )

        self.__load_rpn_from_onnx(os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"))

    @staticmethod
    def download(model_name, path, server_url=None):

        if server_url is None and model_name not in [
            "crowdhuman_dla34",
            "fairmot_dla34",
        ]:
            raise ValueError("Unknown model_name: " + model_name)

        os.makedirs(path, exist_ok=True)

        if server_url is None:
            server_url = os.path.join(
                OPENDR_SERVER_URL, "perception", "object_tracking_2d",
                "fair_mot"
            )

        url = os.path.join(
            server_url, model_name
        )

        model_dir = os.path.join(path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        urlretrieve(os.path.join(
            url, model_name + ".json"
        ), os.path.join(
            model_dir, model_name + ".json"
        ))

        try:
            urlretrieve(os.path.join(
                url, model_name + ".pth"
            ), os.path.join(
                model_dir, model_name + ".pth"
            ))
        except Exception:
            urlretrieve(os.path.join(
                url, model_name + ".tckpt"
            ), os.path.join(
                model_dir, model_name + ".pth"
            ))

        print("Downloaded model", model_name, "to", model_dir)

        return model_dir

    def __convert_to_onnx(self, input_shape, output_name, do_constant_folding=False, verbose=False):
        inp = torch.randn(input_shape).to(self.device)
        input_names = ["data"]
        output_names = self.heads.keys()

        torch.onnx.export(
            self.model, inp, output_name, verbose=verbose, enable_onnx_checker=True,
            do_constant_folding=do_constant_folding, input_names=input_names, output_names=output_names
        )

    def __load_from_onnx(self, path):
        """
        This method loads an ONNX model from the path provided into an onnxruntime inference session.

        :param path: path to ONNX model
        :type path: str
        """
        self.model.rpn_ort_session = ort.InferenceSession(path)

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

    def __load_from_pth(self, model, path, use_original_dict=False):
        all_params = torch.load(path, map_location=self.device)
        model.load_state_dict(all_params if use_original_dict else all_params["state_dict"])

    def _prepare_datasets(
        self,
        dataset,
        val_dataset,
        train_split_paths,
        val_split_paths,
        require_dataset=True,
        require_val_dataset=True,
    ):

        input_dataset_iterator = None
        eval_dataset_iterator = None

        if isinstance(dataset, ExternalDataset):

            dataset_path = dataset.path
            if dataset.dataset_type.lower() != "mot":
                raise ValueError(
                    "ExternalDataset (" + str(dataset) +
                    ") is given as a dataset, but it is not a MOT dataset")

            transforms = T.Compose([T.ToTensor()])
            input_dataset_iterator = JointDataset(
                dataset_path,
                train_split_paths,
                down_ratio=self.down_ratio,
                max_objects=self.max_objs,
                ltrb=self.ltrb,
                mse_loss=self.mse_loss,
                augment=False, transforms=transforms,
            )
        elif isinstance(dataset, DatasetIterator):
            input_dataset_iterator = MappedDatasetIterator(
                dataset,
                lambda d: process_dataset(
                    d[0], d[1], self.ltrb, self.down_ratio,
                    self.max_objs, self.num_classes, self.mse_loss
                )
            )
        else:
            if require_dataset or dataset is not None:
                raise ValueError(
                    "dataset parameter should be an ExternalDataset or a DatasetIterator"
                )

        if isinstance(val_dataset, ExternalDataset):

            val_dataset_path = val_dataset.path
            if val_dataset.dataset_type.lower() != "mot":
                raise ValueError(
                    "ExternalDataset (" + str(val_dataset) +
                    ") is given as a val_dataset, but it is not a MOT dataset"
                )

            eval_dataset_iterator = RawMotDatasetIterator(
                val_dataset_path,
                val_split_paths,
                down_ratio=self.down_ratio,
                max_objects=self.max_objs,
                ltrb=self.ltrb,
                mse_loss=self.mse_loss,
            )

        elif isinstance(val_dataset, DatasetIterator):
            eval_dataset_iterator = val_dataset
        elif val_dataset is None:
            if isinstance(dataset, ExternalDataset):
                val_dataset_path = dataset.path
                if dataset.dataset_type.lower() != "mot":
                    raise ValueError(
                        "ExternalDataset (" + str(dataset) +
                        ") is given as a dataset, but it is not a MOT dataset"
                    )

                eval_dataset_iterator = RawMotDatasetIterator(
                    val_dataset_path,
                    val_split_paths,
                    down_ratio=self.down_ratio,
                    max_objects=self.max_objs,
                    ltrb=self.ltrb,
                    mse_loss=self.mse_loss,
                )

            elif require_val_dataset:
                raise ValueError(
                    "val_dataset is None and can't be derived from" +
                    " the dataset object because the dataset is not an ExternalDataset"
                )
            else:
                eval_dataset_iterator = input_dataset_iterator
        else:
            raise ValueError(
                "val_dataset parameter should be an ExternalDataset or a DatasetIterator or None"
            )

        return input_dataset_iterator, eval_dataset_iterator

    def __create_model(self):

        heads = {
            'hm': self.num_classes,
            'wh': 2 if not self.ltrb else 4,
            'id': self.reid_dim
        }
        if self.reg_offset:
            heads.update({'reg': 2})

        self.heads = heads

        self.model = create_model(self.backbone, heads, self.head_conv)
        self.model.to(self.device)
        self.model.ort_session = None
        self.model.heads_names = heads.keys()

        self.model_optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

        self.tracker = JDETracker(
            self.model,
            self.threshold,
            self.track_buffer,
            self.max_objs,
            self.image_mean,
            self.image_std,
            self.down_ratio,
            self.num_classes,
            self.reg_offset,
            self.ltrb,
            self.frame_rate,
        )

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

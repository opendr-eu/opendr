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
import onnxruntime as ort
from PIL import Image as PilImage
from torchvision.transforms import transforms as T
from opendr.engine.learners import Learner
from opendr.engine.datasets import DatasetIterator, ExternalDataset, MappedDatasetIterator
from opendr.perception.object_tracking_2d.logger import Logger
from opendr.perception.object_tracking_2d.datasets.market1501_dataset import Market1501DatasetIterator
from opendr.perception.object_tracking_2d.deep_sort.algorithm.run import train
from opendr.perception.object_tracking_2d.fair_mot.algorithm.run import evaluate
from opendr.perception.object_tracking_2d.deep_sort.algorithm.deep_sort_tracker import DeepSortTracker
from opendr.engine.data import Image, ImageWithDetections
from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve


class ObjectTracking2DDeepSortLearner(Learner):
    def __init__(
        self,
        lr=0.1,
        iters=-1,
        batch_size=4,
        optimizer="sgd",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda",
        max_dist=0.2,
        min_confidence=0.3,
        nms_max_overlap=0.5,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
    ):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(ObjectTracking2DDeepSortLearner, self).__init__(
            lr=lr,
            iters=iters,
            batch_size=batch_size,
            optimizer=optimizer,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            temp_path=temp_path,
            device=device,
        )

        self.max_dist = max_dist
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget

        self.__create_model()
        self.model_optimizer = torch.optim.SGD(
            self.tracker.deepsort.extractor.net.parameters(), lr, momentum=0.9, weight_decay=5e-4
        )

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

        if self.tracker is None:
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

        if self.tracker.deepsort.extractor.net.ort_session is None:
            model_metadata["model_paths"] = [
                folder_name_no_ext + ".pth",
            ]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pth"

            torch.save({
                'state_dict': self.tracker.deepsort.extractor.net.state_dict()
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
            self.__load_from_pth(self.tracker.deepsort.extractor.net, os.path.join(path, metadata["model_paths"][0]))
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
        epochs,
        val_dataset=None,
        val_epochs=-1,
        logging_path=None,
        silent=False,
        verbose=False,
        train_transforms=None,
        val_transforms=None,
    ):

        logger = Logger(silent, verbose, logging_path)

        (
            input_dataset_iterator,
            eval_dataset_iterator,
        ) = self._prepare_datasets(
            dataset,
            val_dataset,
            require_val_dataset=val_epochs > 0,
        )

        checkpoints_path = os.path.join(self.temp_path, "checkpoints")
        if self.checkpoint_after_iter != 0 or self.checkpoint_load_iter != 0:
            os.makedirs(checkpoints_path, exist_ok=True)

        if self.checkpoint_load_iter != 0:
            state_dict = torch.load(f"checkpoint_{self.checkpoint_load_iter}.pth")
            self.tracker.deepsort.extractor.net.load_state_dict(state_dict)

        if train_transforms is None:
            train_transforms = T.Compose(
                [
                    T.RandomCrop((128, 64), padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

        if val_transforms is None:
            val_transforms = T.Compose(
                [
                    T.Resize((128, 64)),
                    T.ToTensor(),
                    T.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            )

        log_interval = 1 if verbose else 50

        result = train(
            self.tracker.deepsort.extractor.net, input_dataset_iterator, epochs,
            self.iters, eval_dataset_iterator, self.model_optimizer,
            self.batch_size, val_epochs, train_transforms, val_transforms,
            self.device, logger.log, log_interval, checkpoints_path, self.checkpoint_after_iter
        )

        logger.close()

        return result

    def eval(
        self,
        dataset,
        logging_path=None,
        silent=False,
        verbose=False,
    ):

        logger = Logger(silent, verbose, logging_path)

        eval_dataset_iterator = self._prepare_eval_dataset(
            dataset,
        )

        result = evaluate(self.infer, eval_dataset_iterator)

        logger.log(Logger.LOG_WHEN_NORMAL, result)

        logger.close()

        return result

    def infer(self, batch, frame_ids=None):

        if self.tracker is None:
            raise ValueError("No model loaded or created")

        is_single_image = False

        if isinstance(batch, ImageWithDetections):
            batch = [batch]
            is_single_image = True
        elif not isinstance(batch, list):
            raise ValueError("Input batch should be an engine.ImageWithDetections or a list of engine.ImageWithDetections")

        if frame_ids is None:
            frame_ids = [-1] * len(batch)
        elif is_single_image:
            frame_ids = [frame_ids]

        results = []

        for image, frame_id in zip(batch, frame_ids):

            result = self.tracker.infer(image, frame_id)
            results.append(result)

        if is_single_image:
            results = results[0]

        return results

    def optimize(self, do_constant_folding=False, img_size=(64, 128)):
        """
        Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """

        if self.tracker.deepsort.extractor.net is None:
            raise UserWarning("No model is loaded, cannot optimize. Load or train a model first.")
        if self.tracker.deepsort.extractor.net.ort_session is not None:
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
            self.__convert_to_onnx(
                input_shape,
                os.path.join(self.temp_path, "onnx_model_temp.onnx"), do_constant_folding
            )

        self.__load_from_onnx(os.path.join(self.temp_path, "onnx_model_temp.onnx"))

    @staticmethod
    def download(model_name, path, server_url=None):

        if server_url is None and model_name not in [
            "deep_sort",
        ]:
            raise ValueError("Unknown model_name: " + model_name)

        os.makedirs(path, exist_ok=True)

        if server_url is None:
            server_url = os.path.join(
                OPENDR_SERVER_URL, "perception", "object_tracking_2d",
                "deep_sort"
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
        output_names = ["output"]

        torch.onnx.export(
            self.tracker.deepsort.extractor.net, inp, output_name, verbose=verbose, enable_onnx_checker=True,
            do_constant_folding=do_constant_folding, input_names=input_names, output_names=output_names,
            dynamic_axes={"data": {0: "batch"}},
        )

    def __load_from_onnx(self, path):
        """
        This method loads an ONNX model from the path provided into an onnxruntime inference session.

        :param path: path to ONNX model
        :type path: str
        """
        self.tracker.deepsort.extractor.net.ort_session = ort.InferenceSession(path)

        # The comments below are the alternative way to use the onnx model, it might be useful in the future
        # depending on how ONNX saving/loading will be implemented across the toolkit.
        # # Load the ONNX model
        # self.tracker.deepsort.extractor.net = onnx.load(path)
        #
        # # Check that the IR is well formed
        # onnx.checker.check_model(self.tracker.deepsort.extractor.net)
        #
        # # Print a human readable representation of the graph
        # onnx.helper.printable_graph(self.tracker.deepsort.extractor.net.graph)

    def __load_from_pth(self, model, path, use_original_dict=False):
        all_params = torch.load(path, map_location=self.device)
        model.load_state_dict(all_params if use_original_dict else all_params["state_dict"])

    def _prepare_datasets(
        self,
        dataset,
        val_dataset,
        require_dataset=True,
        require_val_dataset=True,
    ):

        input_dataset_iterator = None
        eval_dataset_iterator = None

        def map_dataset(dataset):

            def image_to_pil(image: Image):

                return PilImage.fromarray(image.convert('channels_last'))

            return MappedDatasetIterator(dataset, lambda data: (image_to_pil(data[0]), data[1].data))

        if isinstance(dataset, ExternalDataset):

            dataset_path = dataset.path
            if dataset.dataset_type.lower() != "market1501":
                raise ValueError(
                    "ExternalDataset (" + str(dataset) +
                    ") is given as a dataset, but it is not a Market1501 dataset")

            input_dataset_iterator = map_dataset(Market1501DatasetIterator(os.path.join(dataset_path, "bounding_box_train")))
        elif isinstance(dataset, DatasetIterator):
            input_dataset_iterator = map_dataset(dataset)
        else:
            if require_dataset or dataset is not None:
                raise ValueError(
                    "dataset parameter should be an ExternalDataset or a DatasetIterator"
                )

        if isinstance(val_dataset, ExternalDataset):

            val_dataset_path = val_dataset.path
            if val_dataset.dataset_type.lower() != "market1501":
                raise ValueError(
                    "ExternalDataset (" + str(val_dataset) +
                    ") is given as a val_dataset, but it is not a Market1501 dataset"
                )

            eval_dataset_iterator = map_dataset(Market1501DatasetIterator(os.path.join(val_dataset_path, "bounding_box_test")))

        elif isinstance(val_dataset, DatasetIterator):
            eval_dataset_iterator = map_dataset(val_dataset)
        elif val_dataset is None:
            if isinstance(dataset, ExternalDataset):
                val_dataset_path = dataset.path
                if dataset.dataset_type.lower() != "market1501":
                    raise ValueError(
                        "ExternalDataset (" + str(dataset) +
                        ") is given as a dataset, but it is not a Market1501 dataset"
                    )

                eval_dataset_iterator = map_dataset(Market1501DatasetIterator(
                    os.path.join(val_dataset_path, "bounding_box_test")
                ))

            elif require_val_dataset:
                raise ValueError(
                    "val_dataset is None and can't be derived from" +
                    " the dataset object because the dataset is not an ExternalDataset"
                )
            else:
                eval_dataset_iterator = map_dataset(input_dataset_iterator)
        else:
            raise ValueError(
                "val_dataset parameter should be an ExternalDataset or a DatasetIterator or None"
            )

        return input_dataset_iterator, eval_dataset_iterator

    def _prepare_eval_dataset(
        self,
        dataset,
    ):

        eval_dataset_iterator = None

        if isinstance(dataset, DatasetIterator):
            eval_dataset_iterator = dataset
        else:
            raise ValueError(
                "dataset parameter should be a DatasetIterator"
            )

        return eval_dataset_iterator

    def __create_model(self):

        self.tracker = DeepSortTracker(
            max_dist=self.max_dist,
            min_confidence=self.min_confidence,
            nms_max_overlap=self.nms_max_overlap,
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init,
            nn_budget=self.nn_budget,
            device=self.device,
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

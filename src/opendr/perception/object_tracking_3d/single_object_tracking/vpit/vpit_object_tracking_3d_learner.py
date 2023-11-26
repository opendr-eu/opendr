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

import os
import json
import time
import numpy as np
import torch
import ntpath
import shutil
import pathlib
import onnxruntime as ort
from opendr.engine.learners import Learner
from opendr.engine.datasets import (
    DatasetIterator,
    ExternalDataset,
    MappedDatasetIterator,
)
from opendr.perception.object_tracking_3d.datasets.kitti_siamese_tracking import (
    SiameseTrackingDatasetIterator,
    SiameseTripletTrackingDatasetIterator,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.draw import (
    stack_images,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.regressor.vertical import (
    create_vertical_position_regressor,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.load import (
    create_model as second_create_model,
    load_from_checkpoint,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.run import (
    create_lidar_aabb_from_target,
    create_multi_rotate_searches,
    create_pseudo_image_features,
    create_scaled_scores,
    create_target_search_regions,
    displacement_score_to_image_coordinates,
    draw_pseudo_image,
    evaluate,
    image_to_lidar_coordinates,
    infer_create_pseudo_image,
    original_search_size_by_target_size,
    pc_range_by_lidar_aabb,
    select_best_scores_and_search,
    size_with_context,
    tracking_boxes_to_lidar,
    train,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.pytorch.builder import (
    input_reader_builder,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.logger import (
    Logger,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.pytorch.models.tanet import (
    set_tanet_config,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.data.preprocess import (
    _prep_v9,
    _prep_v9_infer,
)
from opendr.perception.object_tracking_3d.single_object_tracking.vpit.second_detector.builder.dataset_builder import (
    create_prep_func,
)
from opendr.engine.target import (
    TrackingAnnotation3D,
    TrackingAnnotation3DList,
)
from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve


class ObjectTracking3DVpitLearner(Learner):
    def __init__(
        self,
        model_config_path,
        lr=0.0001,
        optimizer="adam_optimizer",
        lr_schedule="exponential_decay_learning_rate",
        checkpoint_after_iter=0,
        checkpoint_load_iter=0,
        temp_path="",
        device="cuda:0",
        tanet_config_path=None,
        optimizer_params={"weight_decay": 0.0001},
        lr_schedule_params={
            "decay_steps": 2000,
            "decay_factor": 0.8,
            "staircase": True,
        },
        feature_blocks=1,
        window_influence=0.85,
        score_upscale=8,
        rotation_penalty=0.98,
        rotation_step=0.15,
        rotations_count=3,
        rotation_interpolation=1,
        target_size=[-1, -1],  # [127, 127],
        search_size=[-1, -1],  # [255, 255],
        context_amount=0.25,  # -0.2  # 0.5,
        target_feature_merge_scale=0.0,
        loss_function="bce",  # focal, bce
        r_pos=2,
        augment=False,
        augment_rotation=True,
        train_pseudo_image=False,
        search_type="small",
        target_type="normal",
        bof_mode="none",
        bof_training_steps=2000,
        extrapolation_mode="linear+",  # "none", "linear", "linear+"
        offset_interpolation=0.25,
        vertical_offset_interpolation=1,
        min_top_score=None,
        overwrite_strides=None,
        target_features_mode="init",  # "init", "all", "selected", "last"
        upscaling_mode="none",  # "none", "raw", "processed"
        regress_vertical_position=False,
        regression_training_isolation=False,
        vertical_regressor_type="center_linear",
        vertical_regressor_kwargs={},
        iters=10,
        backbone="pp",
        batch_size=64,
        threshold=0.0,
        scale=1.0,
    ):
        # Pass the shared parameters on super's constructor so they can get initialized as class attributes
        super(ObjectTracking3DVpitLearner, self).__init__(
            lr=lr,
            iters=iters,
            backbone=backbone,
            batch_size=batch_size,
            threshold=threshold,
            scale=scale,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            checkpoint_after_iter=checkpoint_after_iter,
            checkpoint_load_iter=checkpoint_load_iter,
            temp_path=temp_path,
            device=device,
        )

        self.model_config_path = model_config_path
        self.optimizer_params = optimizer_params
        self.lr_schedule_params = lr_schedule_params

        self.model_dir = None
        self.eval_checkpoint_dir = None
        self.infer_point_cloud_mapper = None

        self.window_influence = window_influence
        self.score_upscale = score_upscale
        self.rotation_penalty = rotation_penalty
        self.rotation_step = rotation_step
        self.rotations_count = rotations_count
        self.rotation_interpolation = rotation_interpolation
        self.target_size = np.array(target_size, dtype=np.int32)
        self.search_size = np.array(search_size, dtype=np.int32)
        self.context_amount = context_amount
        self.feature_blocks = feature_blocks
        self.target_feature_merge_scale = target_feature_merge_scale
        self.loss_function = loss_function
        self.r_pos = r_pos
        self.search_type = search_type
        self.target_type = target_type
        self.bof_mode = bof_mode
        self.augment = augment
        self.augment_rotation = augment_rotation
        self.train_pseudo_image = train_pseudo_image
        self.extrapolation_mode = extrapolation_mode
        self.offset_interpolation = offset_interpolation
        self.min_top_score = min_top_score
        self.regress_vertical_position = regress_vertical_position
        self.regression_training_isolation = regression_training_isolation
        self.overwrite_strides = overwrite_strides
        self.target_features_mode = target_features_mode
        self.upscaling_mode = upscaling_mode
        self.vertical_offset_interpolation = vertical_offset_interpolation
        self.vertical_regressor_type = vertical_regressor_type
        self.vertical_regressor_kwargs = vertical_regressor_kwargs
        self.bof_training_steps = bof_training_steps

        if tanet_config_path is not None:
            set_tanet_config(tanet_config_path)

        self.__create_model()
        self._images = {}
        self.fpses = []
        self.times = {
            "pseudo_image": [],
            "pseudo_image/create_prep_func": [],
            "pseudo_image/infer_point_cloud_mapper": [],
            "pseudo_image/merge_second_batch": [],
            "pseudo_image/branch.create_pseudo_image": [],
            "create_multi_rotate_searches": [],
            "create_pseudo_image_features": [],
            "create_scaled_scores": [],
            "select_best_scores_and_search": [],
            "displacement_score_to_image_coordinates": [],
            "target_feature_merge": [],
            "final_result": [],
        }
        self.training_method = "siamese"
        self.extrapolation_direction = None

        self.model.ort_session = None  # ONNX runtime inference session

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

        if self.model is None:
            raise UserWarning("No model is loaded, cannot save.")

        folder_name, _, tail = self.__extract_trailing(
            path
        )  # Extract trailing folder name from path
        # Also extract folder name without any extension if extension is erroneously provided
        folder_name_no_ext = folder_name.split(sep=".")[0]

        # Extract path without folder name, by removing folder name from original path
        path_no_folder_name = "".join(path.rsplit(folder_name, 1))
        # If tail is '', then path was a/b/c/, which leaves a trailing double '/'
        if tail == "":
            path_no_folder_name = path_no_folder_name[0:-1]  # Remove one '/'

        # Create model directory
        new_path = path_no_folder_name + folder_name_no_ext
        os.makedirs(new_path, exist_ok=True)

        model_metadata = {
            "model_paths": [],
            "framework": "pytorch",
            "format": "",
            "has_data": False,
            "inference_params": {},
            "optimized": None,
            "optimizer_info": {},
        }

        if self.model.ort_session is None:
            model_metadata["model_paths"] = [
                folder_name_no_ext + ".pth",
            ]
            model_metadata["optimized"] = False
            model_metadata["format"] = "pth"

            torch.save(
                {"siamese_model": self.model.state_dict()},
                os.path.join(
                    path_no_folder_name,
                    folder_name_no_ext,
                    model_metadata["model_paths"][0],
                ),
            )
            if verbose:
                print("Saved Pytorch models")
        else:
            model_metadata["model_paths"] = [
                folder_name_no_ext + ".onnx",
            ]
            model_metadata["optimized"] = True
            model_metadata["format"] = "onnx"

            # Copy already optimized model from temp path
            shutil.copy2(
                os.path.join(self.temp_path, "onnx_model_rpn_temp.onnx"),
                os.path.join(
                    path_no_folder_name,
                    folder_name_no_ext,
                    model_metadata["model_paths"][2],
                ),
            )
            if verbose:
                print("Saved ONNX model.")

        with open(os.path.join(new_path, folder_name_no_ext + ".json"), "w") as outfile:
            json.dump(model_metadata, outfile)

    def load(self, path, verbose=False, backbone=False, full=False):
        """
        Loads the model from inside the path provided, based on the metadata .json file included.
        :param path: path of the directory the model was saved
        :type path: str
        :param verbose: whether to print success message or not, defaults to 'False'
        :type verbose: bool, optional
        :param backbone: whether the loaded model is a backbone save file (PointPillars, TANet), not a VPIT save file.
        :type backbone: bool, optional
        :param full: whether the loaded model is a full save, or a checkpoint.
        :type full: bool, optional
        """

        target = self.model.branch if backbone else self.model
        state_dict_name = "state_dict" if backbone or full else "siamese_model"
        use_original = backbone

        model_name, _, _ = self.__extract_trailing(
            path
        )  # Trailing folder name from the path provided

        with open(os.path.join(path, model_name + ".json")) as metadata_file:
            metadata = json.load(metadata_file)

        if len(metadata["model_paths"]) == 1:
            self.__load_from_pth(
                target,
                os.path.join(path, metadata["model_paths"][0]),
                use_original,
                state_dict_name,
            )
            if verbose:
                print("Loaded Pytorch model.")
        else:
            self.__load_from_pth(
                self.model.branch.voxel_feature_extractor,
                os.path.join(path, metadata["model_paths"][0]),
            )
            self.__load_from_pth(
                self.model.branch.middle_feature_extractor,
                os.path.join(path, metadata["model_paths"][1]),
            )
            if verbose:
                print("Loaded Pytorch VFE and MFE sub-model.")

            if not metadata["optimized"]:
                self.__load_from_pth(
                    self.model.branch.rpn,
                    os.path.join(path, metadata["model_paths"][2]),
                )
                if verbose:
                    print("Loaded Pytorch RPN sub-model.")
            else:
                self.__load_rpn_from_onnx(
                    os.path.join(path, metadata["model_paths"][2])
                )
                if verbose:
                    print("Loaded ONNX RPN sub-model.")

    def reset(self):
        pass

    def fit(
        self,
        dataset,
        steps=0,
        val_dataset=None,
        refine_weight=2,
        ground_truth_annotations=None,
        logging_path=None,
        silent=False,
        verbose=False,
        model_dir=None,
        image_shape=(1224, 370),
        evaluate=False,
        debug=False,
        load_optimizer=True,
    ):

        logger = Logger(silent, verbose, logging_path)
        display_step = 1 if verbose else 50

        if model_dir is not None:
            model_dir = pathlib.Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir = model_dir

        if self.model_dir is None and (
            self.checkpoint_load_iter != 0 or self.checkpoint_after_iter != 0
        ):
            raise ValueError(
                "Can not use checkpoint_load_iter or checkpoint_after_iter if model_dir is None and load was not called before"
            )

        (
            input_dataset_iterator,
            eval_dataset_iterator,
            ground_truth_annotations,
        ) = self.__prepare_datasets(
            dataset,
            val_dataset,
            self.input_config,
            self.evaluation_input_config,
            self.model_config,
            self.voxel_generator,
            self.target_assigner,
            ground_truth_annotations,
        )

        checkpoints_path = self.model_dir / "checkpoints"
        if self.checkpoint_after_iter != 0 or self.checkpoint_load_iter != 0:
            checkpoints_path.mkdir(exist_ok=True)

        if self.checkpoint_load_iter != 0:
            self.load_from_checkpoint(
                checkpoints_path,
                self.checkpoint_load_iter,
                load_optimizer=load_optimizer,
            )

        train(
            self.model,
            self.input_config,
            self.train_config,
            self.evaluation_input_config,
            self.model_config,
            self.mixed_optimizer,
            self.lr_scheduler,
            self.model_dir,
            self.float_dtype,
            refine_weight,
            self.loss_scale,
            self.class_names,
            self.center_limit_range,
            input_dataset_iterator=input_dataset_iterator,
            eval_dataset_iterator=eval_dataset_iterator,
            gt_annos=ground_truth_annotations,
            log=logger.log,
            checkpoint_after_iter=self.checkpoint_after_iter,
            checkpoints_path=checkpoints_path,
            display_step=display_step,
            device=self.device,
            image_shape=image_shape,
            evaluate=evaluate,
            context_amount=self.context_amount,
            target_size=self.target_size,
            search_size=self.search_size,
            debug=debug,
            train_steps=steps,
            loss_function=self.loss_function,
            r_pos=self.r_pos,
            infer_point_cloud_mapper=self.infer_point_cloud_mapper,
            augment=self.augment,
            augment_rotation=self.augment_rotation,
            search_type=self.search_type,
            target_type=self.target_type,
            training_method=self.training_method,
            train_pseudo_image=self.train_pseudo_image,
            regress_vertical_position=self.regress_vertical_position,
            regression_training_isolation=self.regression_training_isolation,
            overwrite_strides=self.overwrite_strides,
            upscaling_mode=self.upscaling_mode,
            bof_training_steps=0 if self.bof_mode == "none" else self.bof_training_steps,
        )

        logger.close()

    def eval(
        self,
        dataset,
        ground_truth_annotations=None,
        logging_path=None,
        silent=False,
        verbose=False,
        image_shape=(370, 1224),
        count=None,
    ):

        logger = Logger(silent, verbose, logging_path)

        (_, eval_dataset_iterator, ground_truth_annotations,) = self.__prepare_datasets(
            None,
            dataset,
            self.input_config,
            self.evaluation_input_config,
            self.model_config,
            self.voxel_generator,
            self.target_assigner,
            ground_truth_annotations,
            require_dataset=False,
        )

        result = evaluate(
            self.siamese,
            self.evaluation_input_config,
            self.model_config,
            self.mixed_optimizer,
            self.model_dir,
            self.float_dtype,
            self.class_names,
            self.center_limit_range,
            eval_dataset_iterator=eval_dataset_iterator,
            gt_annos=ground_truth_annotations,
            predict_test=False,
            log=logger.log,
            device=self.device,
            image_shape=image_shape,
            count=count,
        )

        logger.close()

        return result

    def create_pseudo_image(self, point_clouds, pc_range):

        if self.model is None:
            raise ValueError("No model loaded or created")

        result = infer_create_pseudo_image(
            self.model.branch,
            point_clouds,
            pc_range,
            self.infer_point_cloud_mapper,
            self.float_dtype,
            times=self.times,
        )

        return result

    def __add_image(self, image, group):
        if group not in self._images:
            self._images[group] = []

        self._images[group].append(image)

    def load_from_checkpoint(self, checkpoints_path, step, load_optimizer=True):
        self.lr_scheduler = load_from_checkpoint(
            self.model,
            self.mixed_optimizer,
            str(checkpoints_path) + f"/checkpoint_{step}.pth",
            self.lr_schedule,
            self.lr_schedule_params,
            self.device,
            load_optimizer=load_optimizer,
        )

    def infer(self, point_cloud, frame=0, id=None, draw=False):

        with torch.no_grad():

            t = time.time()

            net = self.model.branch

            pc_range = pc_range_by_lidar_aabb(self.search_lidar_aabb)
            pseudo_images = self.create_pseudo_image(point_cloud, pc_range)
            pseudo_image = pseudo_images[0]

            t1 = time.time()
            self.times["pseudo_image"].append(t1 - t)

            if draw:
                draw_pi = draw_pseudo_image(
                    pseudo_image.squeeze(axis=0),
                    "./plots/small_pi/" + str(frame) + ".png",
                )
                draw_pseudo_image(
                    pseudo_image.squeeze(axis=0),
                    "./plots/small_pi/a.png",
                )
                self.__add_image(draw_pi, "small_pi")

            t1 = time.time()

            multi_rotate_searches_and_penalties = create_multi_rotate_searches(
                self.search_region,
                self.rotation_penalty,
                self.rotation_step,
                self.rotations_count,
            )

            t2 = time.time()
            self.times["create_multi_rotate_searches"].append(t2 - t1)

            multi_rotate_features_and_searches_and_penalties = []

            for i, (search, penalty) in enumerate(multi_rotate_searches_and_penalties):
                search_features, search_image = create_pseudo_image_features(
                    pseudo_image,
                    search,
                    net,
                    self.search_size,
                    self.context_amount,
                    offset=self.search_region[0],
                )

                multi_rotate_features_and_searches_and_penalties.append(
                    [search_features, search, penalty]
                )

                if draw:
                    draw_search = draw_pseudo_image(
                        search_image.squeeze(axis=0),
                        "./plots/search/" + str(frame) + "_" + str(i) + ".png",
                    )
                    draw_pseudo_image(
                        search_image.squeeze(axis=0),
                        "./plots/search/a.png",
                    )

                    draw_search_feat = draw_pseudo_image(
                        search_features.squeeze(axis=0),
                        "./plots/search_feat/" + str(frame) + "_" + str(i) + ".png",
                    )
                    draw_pseudo_image(
                        search_features.squeeze(axis=0),
                        "./plots/search_feat/a.png",
                    )

                    self.__add_image(draw_search, "search")
                    self.__add_image(draw_search_feat, "search_feat")

            t3 = time.time()
            self.times["create_pseudo_image_features"].append(t3 - t2)

            multi_rotate_scores_searches_penalties_and_features = []

            for i, (
                search_features,
                target,
                penalty,
            ) in enumerate(multi_rotate_features_and_searches_and_penalties):

                target_features_to_compare = [self.init_target_features]

                if self.target_features_mode in ["all", "selected", "last"]:
                    target_features_to_compare = [
                        *target_features_to_compare,
                        *self.lifetime_target_features,
                    ]

                for it, target_features in enumerate(target_features_to_compare):
                    scores, original_scores, penalty_map = create_scaled_scores(
                        target_features,
                        search_features,
                        self.model,
                        self.score_upscale,
                        self.window_influence,
                        self.extrapolation_direction,
                    )
                    multi_rotate_scores_searches_penalties_and_features.append(
                        [scores, target, penalty, search_features]
                    )

                    if draw:
                        draw_scores = draw_pseudo_image(
                            scores.squeeze(axis=0),
                            "./plots/scores/" +
                            str(frame) +
                            "_" +
                            str(i) +
                            "_" +
                            str(it) +
                            ".png",
                        )
                        draw_scores_original = draw_pseudo_image(
                            original_scores.squeeze(axis=0),
                            "./plots/scores_original/" +
                            str(frame) +
                            "_" +
                            str(i) +
                            "_" +
                            str(it) +
                            ".png",
                        )
                        draw_penalty_map = draw_pseudo_image(
                            penalty_map,
                            "./plots/penalty_map/" +
                            str(frame) +
                            "_" +
                            str(i) +
                            "_" +
                            str(it) +
                            ".png",
                        )
                        self.__add_image(draw_scores, "scores")
                        self.__add_image(draw_scores_original, "scores_original")
                        self.__add_image(draw_penalty_map, "penalty_map")

            t4 = time.time()
            self.times["create_scaled_scores"].append(t4 - t3)

            (
                top_scores,
                top_search,
                top_search_features,
            ) = select_best_scores_and_search(
                multi_rotate_scores_searches_penalties_and_features
            )

            t5 = time.time()
            self.times["select_best_scores_and_search"].append(t5 - t4)

            if draw:

                draw_pseudo_image(
                    top_scores.squeeze(axis=0),
                    "./plots/scores/" + str(frame) + "_top.png",
                )

                max_score = torch.max(top_scores)
                max_idx = (top_scores == max_score).nonzero(as_tuple=False)[0][-2:]

                draw_pseudo_image(
                    top_scores.squeeze(axis=0),
                    "./plots/scores/" + str(frame) + "_top_marked.png",
                    [[max_idx.cpu().numpy(), np.array([2, 2]), 0]],
                    [(255, 0, 0)],
                )

            t5 = time.time()

            delta_image, norm_max = displacement_score_to_image_coordinates(
                top_scores,
                self.score_upscale,
                top_search[1],
                top_search[2],
                self.search_size,
            )

            new_angle = top_search[
                2
            ] * self.rotation_interpolation + self.search_region[2] * (
                1 - self.rotation_interpolation
            )

            unreliable = (
                self.min_top_score is not None and norm_max <= self.min_top_score
            )

            if unreliable:
                if self.extrapolation_mode == "linear":
                    delta_image = (
                        self.extrapolation_direction / self.offset_interpolation
                    )
                if self.extrapolation_mode == "linear+":
                    delta_image = (
                        self.extrapolation_direction / self.offset_interpolation
                    )
                elif self.extrapolation_mode == "none":
                    delta_image = np.array([0, 0], dtype=delta_image.dtype)
                else:
                    raise ValueError()

                new_angle = self.search_region[2]

            delta_image = delta_image[[1, 0]]
            delta_image *= self.offset_interpolation
            center_image = self.search_region[0] + delta_image

            new_target = [center_image, self.init_target[1], new_angle]
            new_search = [
                center_image,
                original_search_size_by_target_size(new_target[1], self.search_type),
                new_angle,
            ]

            if not unreliable:
                if self.extrapolation_mode == "linear":
                    new_search[0] += delta_image
                    self.extrapolation_direction = delta_image[[1, 0]]
                elif self.extrapolation_mode == "linear+":
                    new_search[0] += new_target[0] - self.last_target[0]
                    self.extrapolation_direction = (
                        new_target[0] - self.last_target[0]
                    )[[1, 0]]
                elif self.extrapolation_mode == "none":
                    pass
                else:
                    raise ValueError()

            self.search_region = new_search
            self.last_target = new_target

            t6 = time.time()
            self.times["displacement_score_to_image_coordinates"].append(t6 - t5)

            vertical_position = self.init_label.location[-1]

            create_target_features = (
                self.target_feature_merge_scale > 0 or
                self.regress_vertical_position or
                self.target_features_mode in ["all", "selected", "last"]
            )

            if create_target_features and not unreliable:
                target_features, target_image = create_pseudo_image_features(
                    pseudo_image,
                    new_target,
                    net,
                    self.target_size,
                    self.context_amount,
                    offset=target[0],
                )

                if self.target_feature_merge_scale:
                    self.init_target_features = (
                        self.init_target_features *
                        (1 - self.target_feature_merge_scale) +
                        target_features * self.target_feature_merge_scale
                    )

                if self.regress_vertical_position:
                    vertical_position = (
                        (
                            np.mean(self.model.branch.point_cloud_range[[2, 5]]) +
                            self.model.vertical_position_regressor(target_features)
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )

                if self.target_features_mode in ["all", "selected", "last"]:

                    selected = True

                    if self.target_features_mode == "selected":
                        selected = True

                    if selected:
                        self.lifetime_target_features.append(target_features)

                    if self.target_features_mode == "last":
                        self.lifetime_target_features = [
                            self.lifetime_target_features[-1]
                        ]

                if draw:
                    draw_target_feat_full = draw_pseudo_image(
                        self.init_target_features.squeeze(axis=0),
                        "./plots/target_feat/" + str(frame) + "_target_feat_full.png",
                    )
                    draw_target_feat_current_frame = draw_pseudo_image(
                        target_features.squeeze(axis=0),
                        "./plots/scores/" +
                        str(frame) +
                        "_target_feat_current_frame.png",
                    )

                    draw_target_image = draw_pseudo_image(
                        target_image.squeeze(axis=0),
                        "./plots/scores/" +
                        str(frame) +
                        "_target_image_current_frame.png",
                    )
                    self.__add_image(draw_target_feat_full, "target_feat_full")
                    self.__add_image(
                        draw_target_feat_current_frame, "target_feat_current_frame"
                    )
                    self.__add_image(draw_target_image, "target_image_current_frame")

            vertical_position = (
                vertical_position * self.vertical_offset_interpolation +
                self.last_vertical_position * (1 - self.vertical_offset_interpolation)
            )
            self.last_vertical_position = vertical_position

            t7 = time.time()
            self.times["target_feature_merge"].append(t7 - t6)

            location_lidar, size_lidar = image_to_lidar_coordinates(
                new_target[0], new_target[1], net.voxel_size, net.bv_range
            )

            search_size_with_context = size_with_context(
                self.search_region[1], self.context_amount
            )

            self.search_lidar_aabb = create_lidar_aabb_from_target(
                [self.search_region[0], search_size_with_context, target[2]],
                net.voxel_size,
                net.bv_range,
                net.point_cloud_range[[2, 5]],
            )

            result = TrackingAnnotation3DList(
                [
                    TrackingAnnotation3D(
                        self.init_label.name,
                        0,
                        0,
                        None,
                        None,
                        location=np.array([*location_lidar, vertical_position]),
                        dimensions=self.init_label.dimensions,
                        rotation_y=new_target[2],
                        id=self.init_label.id if id is None else id,
                        score=1,
                        frame=frame,
                    )
                ]
            )

            t8 = time.time()
            self.times["final_result"].append(t7 - t6)

            fps = 1 / (t8 - t)

            self.fpses.append(fps)

            if draw:
                image = self._images["search"][-1]
                image = stack_images(
                    [image, self._images["scores_original"][-1]], "horizontal"
                )

                images_feat = [self._images["search_feat"][-1]]
                if "target_feat_current_frame" in self._images:
                    images_feat.append(self._images["target_feat_current_frame"][-1])
                    images_feat.append(self._images["target_feat_full"][-1])
                    images_feat.append(self._images["target_image_current_frame"][-1])

                image = stack_images(
                    [image, stack_images(images_feat, "horizontal")], "vertical"
                )
                image = stack_images(
                    [
                        image,
                        self._images["small_pi"][-1],
                        self._images["penalty_map"][-1],
                    ],
                    "vertical",
                )
                self.__add_image(image, "summary")

                draw_pseudo_image(
                    image,
                    "./plots/summary/" + str(frame) + "_" + str(i) + ".png",
                )

                draw_pseudo_image(
                    image,
                    "./plots/summary/a" + ".png",
                )

            return result

    def init(self, point_cloud, label_lidar, draw=False, clear_metrics=False):

        self.model.eval()
        self._images = {}

        if clear_metrics:
            self.fpses = []

            self.times = {
                "pseudo_image": [],
                "pseudo_image/create_prep_func": [],
                "pseudo_image/infer_point_cloud_mapper": [],
                "pseudo_image/merge_second_batch": [],
                "pseudo_image/branch.create_pseudo_image": [],
                "create_multi_rotate_searches": [],
                "create_pseudo_image_features": [],
                "create_scaled_scores": [],
                "select_best_scores_and_search": [],
                "displacement_score_to_image_coordinates": [],
                "target_feature_merge": [],
                "final_result": [],
            }

        self.init_label = label_lidar

        label_lidar_kitti = label_lidar.kitti()

        dims = label_lidar_kitti["dimensions"]
        locs = label_lidar_kitti["location"]
        rots = label_lidar_kitti["rotation_y"][0]

        box_lidar = np.concatenate([locs, dims, rots[..., np.newaxis]], axis=1)

        net = self.model.branch

        batch_targets, batch_searches = create_target_search_regions(
            net.bv_range,
            net.voxel_size,
            boxes_lidar=box_lidar.reshape(1, *box_lidar.shape),
            augment=False,
            search_type=self.search_type,
            target_type=self.target_type,
        )

        target = batch_targets[0][0]
        search = batch_searches[0][0]

        init_size_with_context = size_with_context(target[1], self.context_amount)

        init_lidar_aabb = create_lidar_aabb_from_target(
            [target[0], init_size_with_context, target[2]],
            net.voxel_size,
            net.bv_range,
            net.point_cloud_range[[2, 5]],
        )

        pc_range = pc_range_by_lidar_aabb(init_lidar_aabb)
        pseudo_images = self.create_pseudo_image(point_cloud, pc_range)
        pseudo_image = pseudo_images[0]

        self.init_target_features, init_image = create_pseudo_image_features(
            pseudo_image,
            target,
            net,
            self.target_size,
            self.context_amount,
            offset=target[0],
        )

        self.lifetime_target_features = []

        if draw:
            draw_pi = draw_pseudo_image(
                pseudo_image.squeeze(0),
                "./plots/init/pseudo_image.png",
            )
            draw_init = draw_pseudo_image(
                init_image.squeeze(0),
                "./plots/init/image.png",
            )
            draw_feat = draw_pseudo_image(
                self.init_target_features.squeeze(0),
                "./plots/init/target_features.png",
            )

            image = stack_images([draw_pi, draw_init], "vertical")
            image = stack_images([image, draw_feat], "vertical")

            draw_pseudo_image(
                image,
                "./plots/init/all.png",
            )

        self.search_region = search
        self.init_target = target
        self.last_target = target

        search_size_with_context = size_with_context(
            self.search_region[1], self.context_amount
        )

        self.search_lidar_aabb = create_lidar_aabb_from_target(
            [self.search_region[0], search_size_with_context, target[2]],
            net.voxel_size,
            net.bv_range,
            net.point_cloud_range[[2, 5]],
        )

        self.last_vertical_position = self.init_label.location[-1]

    def optimize(self, do_constant_folding=False):
        """
        Not Implemented. Optimize method converts the model to ONNX format and saves the
        model in the parent directory defined by self.temp_path. The ONNX model is then loaded.
        :param do_constant_folding: whether to optimize constants, defaults to 'False'
        :type do_constant_folding: bool, optional
        """
        raise NotImplementedError()

    @staticmethod
    def download(model_name, path, server_url=None):

        if server_url is not None and model_name not in [
            "pointpillars_car_xyres_16",
            "pointpillars_ped_cycle_xyres_16",
            "tanet_car_xyres_16",
            "tanet_ped_cycle_xyres_16",
        ]:
            raise ValueError("Unknown model_name: " + model_name)

        os.makedirs(path, exist_ok=True)

        if server_url is None:
            server_url = os.path.join(
                OPENDR_SERVER_URL,
                "perception",
                "object_detection_3d",
                "voxel_object_detection_3d",
            )

        url = os.path.join(server_url, model_name)

        model_dir = os.path.join(path, model_name)
        os.makedirs(model_dir, exist_ok=True)

        urlretrieve(
            os.path.join(url, model_name + ".json"),
            os.path.join(model_dir, model_name + ".json"),
        )

        try:
            urlretrieve(
                os.path.join(url, model_name + ".pth"),
                os.path.join(model_dir, model_name + ".pth"),
            )
        except Exception:
            urlretrieve(
                os.path.join(url, model_name + ".tckpt"),
                os.path.join(model_dir, model_name + ".pth"),
            )

        print("Downloaded model", model_name, "to", model_dir)

        return model_dir

    def fps(self):
        return -1 if len(self.fpses) <= 0 else (sum(self.fpses) / len(self.fpses))

    def __convert_rpn_to_onnx(
        self,
        input_shape,
        has_refine,
        output_name,
        do_constant_folding=False,
        verbose=False,
    ):
        inp = torch.randn(input_shape).to(self.device)
        input_names = ["data"]
        output_names = ["box_preds", "cls_preds", "dir_cls_preds"]

        if has_refine:
            output_names.append("Refine_loc_preds")
            output_names.append("Refine_cls_preds")
            output_names.append("Refine_dir_preds")

        torch.onnx.export(
            self.model.rpn,
            inp,
            output_name,
            verbose=verbose,
            enable_onnx_checker=True,
            do_constant_folding=do_constant_folding,
            input_names=input_names,
            output_names=output_names,
        )

    def __load_rpn_from_onnx(self, path):
        """
        This method loads an ONNX model from the path provided into an onnxruntime inference session.

        :param path: path to ONNX model
        :type path: str
        """
        self.model.ort_session = ort.InferenceSession(path)

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

    def __load_from_pth(
        self,
        model,
        path,
        use_original_dict=False,
        state_dict_name="state_dict",
    ):
        all_params = torch.load(path, map_location=self.device)
        model.load_state_dict(
            all_params if use_original_dict else all_params[state_dict_name],
            strict=self.bof_mode == "none" and self.upscaling_mode == "none",
        )

    def __prepare_datasets(
        self,
        dataset,
        val_dataset,
        input_cfg,
        eval_input_cfg,
        model_cfg,
        voxel_generator,
        target_assigner,
        gt_annos,
        require_dataset=True,
    ):
        def create_map_point_cloud_dataset_func(is_training):

            prep_func = create_prep_func(
                input_cfg if is_training else eval_input_cfg,
                model_cfg,
                is_training,
                voxel_generator,
                target_assigner,
                use_sampler=False,
            )

            def map(data_target):

                point_cloud_with_calibration, target = data_target
                point_cloud = point_cloud_with_calibration.data
                calib = point_cloud_with_calibration.calib

                annotation = target.kitti()

                example = _prep_v9(point_cloud, calib, prep_func, annotation)

                if not is_training:
                    example["annos"] = annotation

                if point_cloud_with_calibration.image_shape is not None:
                    example["image_shape"] = point_cloud_with_calibration.image_shape

                return example

            return map

        def create_map_siamese_dataset_func(is_training):
            def map(data_target):

                (
                    target_point_cloud_calib,
                    search_point_cloud_calib,
                    target_label,
                    search_label,
                ) = data_target
                target_point_cloud = target_point_cloud_calib.data
                search_point_cloud = search_point_cloud_calib.data
                calib = target_point_cloud_calib.calib

                target_label_lidar = tracking_boxes_to_lidar(target_label, calib)
                search_label_lidar = tracking_boxes_to_lidar(search_label, calib)

                target_label_lidar_kitti = target_label_lidar.kitti()
                search_label_lidar_kitti = search_label_lidar.kitti()

                del target_label_lidar_kitti["name"]
                del search_label_lidar_kitti["name"]

                return (
                    target_point_cloud,
                    search_point_cloud,
                    target_label_lidar_kitti,
                    search_label_lidar_kitti,
                )

            return map

        def create_map_siamese_triplet_dataset_func(is_training):
            def map(data_target):

                (
                    target_point_cloud_calib,
                    search_point_cloud_calib,
                    other_point_cloud_calib,
                    target_label,
                    search_label,
                    other_label,
                ) = data_target
                target_point_cloud = target_point_cloud_calib.data
                search_point_cloud = search_point_cloud_calib.data
                other_point_cloud = other_point_cloud_calib.data
                calib = target_point_cloud_calib.calib

                target_label_lidar = tracking_boxes_to_lidar(target_label, calib)
                search_label_lidar = tracking_boxes_to_lidar(search_label, calib)
                other_label_lidar = tracking_boxes_to_lidar(other_label, calib)

                target_label_lidar_kitti = target_label_lidar.kitti()
                search_label_lidar_kitti = search_label_lidar.kitti()
                other_label_lidar_kitti = other_label_lidar.kitti()

                del target_label_lidar_kitti["name"]
                del search_label_lidar_kitti["name"]
                del other_label_lidar_kitti["name"]

                return (
                    target_point_cloud,
                    search_point_cloud,
                    other_point_cloud,
                    target_label_lidar_kitti,
                    search_label_lidar_kitti,
                    other_label_lidar_kitti,
                )

            return map

        input_dataset_iterator = None
        eval_dataset_iterator = None

        if isinstance(dataset, ExternalDataset):

            if dataset.dataset_type.lower() != "kitti":
                raise ValueError(
                    "ExternalDataset (" +
                    str(dataset) +
                    ") is given as a dataset, but it is not a KITTI dataset"
                )

            dataset_path = dataset.path
            input_cfg.kitti_info_path = dataset_path + "/" + input_cfg.kitti_info_path
            input_cfg.kitti_root_path = dataset_path + "/" + input_cfg.kitti_root_path
            input_cfg.record_file_path = dataset_path + "/" + input_cfg.record_file_path
            input_cfg.database_sampler.database_info_path = (
                dataset_path + "/" + input_cfg.database_sampler.database_info_path
            )

            input_dataset_iterator = input_reader_builder.build(
                input_cfg,
                model_cfg,
                training=True,
                voxel_generator=voxel_generator,
                target_assigner=target_assigner,
                model=self.model,
            )
        elif isinstance(dataset, SiameseTrackingDatasetIterator):
            input_dataset_iterator = MappedDatasetIterator(
                dataset,
                create_map_siamese_dataset_func(True),
            )
            self.training_method = "siamese"
        elif isinstance(dataset, SiameseTripletTrackingDatasetIterator):
            input_dataset_iterator = MappedDatasetIterator(
                dataset,
                create_map_siamese_triplet_dataset_func(True),
            )
            self.training_method = "siamese_triplet"
        elif isinstance(dataset, DatasetIterator):
            input_dataset_iterator = MappedDatasetIterator(
                dataset,
                create_map_siamese_dataset_func(True),
            )
        else:
            if require_dataset or dataset is not None:
                raise ValueError(
                    "dataset parameter should be an ExternalDataset or a DatasetIterator"
                )

        if isinstance(val_dataset, ExternalDataset):

            val_dataset_path = val_dataset.path
            if val_dataset.dataset_type.lower() != "kitti":
                raise ValueError(
                    "ExternalDataset (" +
                    str(val_dataset) +
                    ") is given as a val_dataset, but it is not a KITTI dataset"
                )

            eval_input_cfg.kitti_info_path = (
                val_dataset_path + "/" + eval_input_cfg.kitti_info_path
            )
            eval_input_cfg.kitti_root_path = (
                val_dataset_path + "/" + eval_input_cfg.kitti_root_path
            )
            eval_input_cfg.record_file_path = (
                val_dataset_path + "/" + eval_input_cfg.record_file_path
            )
            eval_input_cfg.database_sampler.database_info_path = (
                val_dataset_path +
                "/" +
                eval_input_cfg.database_sampler.database_info_path
            )

            eval_dataset_iterator = input_reader_builder.build(
                eval_input_cfg,
                model_cfg,
                training=False,
                voxel_generator=voxel_generator,
                target_assigner=target_assigner,
                model=self.model,
            )

            if gt_annos is None:
                gt_annos = [
                    info["annos"] for info in eval_dataset_iterator.dataset.kitti_infos
                ]

        elif isinstance(val_dataset, SiameseTrackingDatasetIterator):
            eval_dataset_iterator = MappedDatasetIterator(
                val_dataset,
                create_map_siamese_dataset_func(True),
            )
        elif isinstance(val_dataset, SiameseTripletTrackingDatasetIterator):
            eval_dataset_iterator = MappedDatasetIterator(
                val_dataset,
                create_map_siamese_triplet_dataset_func(True),
            )
        elif isinstance(val_dataset, DatasetIterator):
            eval_dataset_iterator = MappedDatasetIterator(
                val_dataset,
                create_map_point_cloud_dataset_func(False),
            )
        elif val_dataset is None:
            if isinstance(dataset, ExternalDataset):
                dataset_path = dataset.path
                if dataset.dataset_type.lower() != "kitti":
                    raise ValueError(
                        "ExternalDataset (" +
                        str(dataset) +
                        ") is given as a dataset, but it is not a KITTI dataset"
                    )

                eval_input_cfg.kitti_info_path = (
                    dataset_path + "/" + eval_input_cfg.kitti_info_path
                )
                eval_input_cfg.kitti_root_path = (
                    dataset_path + "/" + eval_input_cfg.kitti_root_path
                )
                eval_input_cfg.record_file_path = (
                    dataset_path + "/" + eval_input_cfg.record_file_path
                )
                eval_input_cfg.database_sampler.database_info_path = (
                    dataset_path +
                    "/" +
                    eval_input_cfg.database_sampler.database_info_path
                )

                eval_dataset_iterator = input_reader_builder.build(
                    eval_input_cfg,
                    model_cfg,
                    training=False,
                    voxel_generator=voxel_generator,
                    target_assigner=target_assigner,
                    model=self.model,
                )

                if gt_annos is None:
                    gt_annos = [
                        info["annos"]
                        for info in eval_dataset_iterator.dataset.kitti_infos
                    ]
            elif isinstance(dataset, DatasetIterator):
                eval_dataset_iterator = MappedDatasetIterator(
                    dataset,
                    create_map_siamese_dataset_func(True),
                )
            else:
                raise ValueError(
                    "val_dataset is None and can't be derived from" +
                    " the dataset object because the dataset is not an ExternalDataset"
                )
        else:
            raise ValueError(
                "val_dataset parameter should be an ExternalDataset or a DatasetIterator or None"
            )

        return input_dataset_iterator, eval_dataset_iterator, gt_annos

    def __create_model(self):
        (
            model,
            input_config,
            train_config,
            evaluation_input_config,
            model_config,
            voxel_generator,
            target_assigner,
            mixed_optimizer,
            lr_scheduler,
            float_dtype,
            loss_scale,
            class_names,
            center_limit_range,
        ) = second_create_model(
            self.model_config_path,
            device=self.device,
            optimizer_name=self.optimizer,
            optimizer_params=self.optimizer_params,
            lr=self.lr,
            feature_blocks=self.feature_blocks,
            lr_schedule_name=self.lr_schedule,
            lr_schedule_params=self.lr_schedule_params,
            loss_function=self.loss_function,
            bof_mode=self.bof_mode,
            overwrite_strides=self.overwrite_strides,
            upscaling_mode=self.upscaling_mode,
        )

        self.model = model
        self.input_config = input_config
        self.train_config = train_config
        self.evaluation_input_config = evaluation_input_config
        self.model_config = model_config
        self.train_config = train_config
        self.voxel_generator = voxel_generator
        self.target_assigner = target_assigner
        self.mixed_optimizer = mixed_optimizer
        self.lr_scheduler = lr_scheduler

        self.float_dtype = float_dtype
        self.loss_scale = loss_scale
        self.class_names = class_names
        self.center_limit_range = center_limit_range

        prep_func = create_prep_func(
            self.input_config,
            self.model_config,
            False,
            self.voxel_generator,
            self.target_assigner,
            use_sampler=False,
            max_number_of_voxels=2000,
        )

        def infer_point_cloud_mapper(x, pc_range):
            return _prep_v9_infer(x, prep_func, pc_range)

        self.infer_point_cloud_mapper = infer_point_cloud_mapper

        if self.regress_vertical_position:
            self.model.vertical_position_regressor = create_vertical_position_regressor(
                self.model.branch.rpn.final_filters,
                self.vertical_regressor_type,
                **self.vertical_regressor_kwargs,
            )

            self.model.vertical_position_regressor.to(self.model.branch.device)
            self.model.vertical_criterion = torch.nn.SmoothL1Loss()

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

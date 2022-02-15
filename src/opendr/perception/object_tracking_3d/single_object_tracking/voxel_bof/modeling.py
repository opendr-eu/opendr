import sys
import os
import torch
import fire
from multiprocessing import Process, set_start_method, get_context

from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.test import (
    test_pp_siamese_fit,
    test_pp_siamese_fit_siamese_training,
    test_rotated_pp_siamese_eval,
    test_rotated_pp_siamese_infer,
)

default_track_ids = [
    # "0000",
    # "0001",
    # "0002",
    # "0003",
    # "0004",
    # "0005",
    # "0006",
    # "0007",
    # "0008",
    # "0009",
    # "0010",
    # "0011",
    # "0012",
    # "0013",
    # "0014",
    # "0015",
    # "0016",
    "0017",
    "0018",
    # "0019",
    # "0020",
]


class Model:
    def __init__(
        self,
        model_name,
        train_steps=200000,
        save_step=2000,
        loads=[-1, -0.5, 2000, 0.2],
        track_ids=default_track_ids,
        decay_steps=2000,
        iou_min=0.5,
        training_method="siamese",
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self.train_steps = train_steps
        self.save_step = save_step
        self.loads = loads
        self.track_ids = track_ids
        self.iou_min = iou_min
        self.kwargs = kwargs
        self.training_method = training_method
        self.kwargs["lr_schedule_params"] = {
            "decay_steps": decay_steps,
            "decay_factor": 0.8,
            "staircase": True,
        }

        print("self.kwargs", self.kwargs)

    def train(
        self, steps, device, eval_after=True, eval_kwargs={"default": {}},
    ):

        steps = steps - (steps % self.save_step)

        print("Sending on", device)

        last_checkpoint_path = (
            "./temp/" + self.model_name + "/checkpoints" + f"/checkpoint_{steps}.pth"
        )

        if os.path.exists(last_checkpoint_path):
            print("The model is already trained")
        else:
            if self.training_method == "detection":
                test_pp_siamese_fit(
                    self.model_name,
                    0,
                    steps,
                    False,
                    checkpoint_after_iter=self.save_step,
                    device=device,
                    **self.kwargs,
                )
            elif self.training_method == "siamese":
                test_pp_siamese_fit_siamese_training(
                    self.model_name,
                    0,
                    steps,
                    False,
                    checkpoint_after_iter=self.save_step,
                    device=device,
                    **self.kwargs,
                )
            else:
                raise ValueError()

        if eval_after:
            return self.eval(
                steps, self.loads, device, self.track_ids, eval_kwargs=eval_kwargs,
            )

        return {}

    def eval(
        self,
        train_steps,
        loads,
        device,
        track_ids=default_track_ids,
        eval_kwargs={"default": {}},
    ):

        results = {}

        for load in loads:

            if abs(load) < self.save_step:
                load = train_steps * load

            if load == -train_steps:
                load = train_steps

            if load < 0:
                load = train_steps + load

            load = int(load)
            load = load - (load % self.save_step)

            for id, kwargs in eval_kwargs.items():

                result = test_rotated_pp_siamese_eval(
                    self.model_name,
                    load,
                    False,
                    self.iou_min,
                    tracks=track_ids,
                    device=device,
                    eval_id=id,
                    **self.kwargs,
                    **kwargs,
                )
                results[str(load) + "_" + str(id)] = result

        return results

    def eval_and_train(self, device, eval_kwargs={"default": {}}):
        return self.train(self.train_steps, device, True, eval_kwargs=eval_kwargs)


def run_all(device_id=0, total_devices=4):
    def create_eval_kwargs():
        params = {
            "window_influence": [0.15, 0.25, 0.05],
            "score_upscale": [16],
            "rotation_penalty": [0.98, 0.96],
            "rotation_step": [0.15, 0.1, 0.075],
            "rotations_count": [3, 5],
        }

        results = {}

        for window_influence in params["window_influence"]:
            for score_upscale in params["score_upscale"]:
                for rotation_penalty in params["rotation_penalty"]:
                    for rotation_step in params["rotation_step"]:
                        for rotations_count in params["rotations_count"]:
                            name = (
                                str(rotations_count).replace(".", "")
                                + "r"
                                + str(rotation_step).replace(".", "")
                                + "-rp"
                                + str(rotation_penalty).replace(".", "")
                                + "su"
                                + str(score_upscale).replace(".", "")
                            )

                            results[name] = {
                                "window_influence": window_influence,
                                "score_upscale": score_upscale,
                                "rotation_penalty": rotation_penalty,
                                "rotation_step": rotation_step,
                                "rotations_count": rotations_count,
                            }
        return results

    eval_kwargs = create_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [3, 2, 1]:
            for size in [1, -1]:
                for context_amount in [0.5, -0.5, -0.2, 0.2, 0]:

                    target_size = [127, 127] if size == 1 else [-1, -1]
                    search_size = [255, 255] if size == 1 else [-1, -1]

                    name = (
                        "0-b"
                        + str(feature_blocks)
                        + ("-us" if size == 1 else "-os")
                        + "-c"
                        + str(context_amount).replace(".", "")
                    )
                    result.append(
                        (
                            Model(
                                name,
                                feature_blocks=feature_blocks,
                                target_size=target_size,
                                search_size=search_size,
                                context_amount=context_amount,
                            ),
                            eval_kwargs,
                        )
                    )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def collect_results():

    models_path = "./temp/"

    models = os.listdir(models_path)

    results = []

    for model in models:
        files = [f for f in os.listdir(models_path + "/" + model) if "results_" in f]

        for file in files:
            with open(models_path + "/" + model + "/" + file, "r") as f:
                str_values = f.readlines()

                values = {}

                for s in str_values:
                    key, value = s.split(" = ")
                    values[key] = value

                result = [
                    model + "_" + file,
                    float(values["total_mean_iou3d"]),
                    float(
                        values["total_precision"] if "total_precision" in values else -1
                    ),
                    float(values["total_success"] if "total_success" in values else -1),
                    float(values["fps"] if "fps" in values else -1),
                ]
                results.append(result)

    results = sorted(results, key=lambda x: x[3])
    for name, iou3d, precision, success, fps in results:
        print(
            name, "precision", precision, "success", success, "iou3d", iou3d, "fps", fps
        )

    with open("modeling.txt", "w") as f:
        for name, iou3d, precision, success, fps in results:
            print(
                name,
                "precision",
                precision,
                "success",
                success,
                "iou3d",
                iou3d,
                "fps",
                fps,
                file=f,
            )


def run_selected(device_id=0, total_devices=4):
    def create_eval_kwargs():
        params = {
            "window_influence": [0.15, 0.25, 0.05],
            "score_upscale": [16],
            "rotation_penalty": [0.98, 0.96],
            "rotation_step": [0.15, 0.1, 0.075],
            "rotations_count": [3, 5],
        }

        results = {}

        for window_influence in params["window_influence"]:
            for score_upscale in params["score_upscale"]:
                for rotation_penalty in params["rotation_penalty"]:
                    for rotation_step in params["rotation_step"]:
                        for rotations_count in params["rotations_count"]:
                            name = (
                                str(rotations_count).replace(".", "")
                                + "r"
                                + str(rotation_step).replace(".", "")
                                + "-rp"
                                + str(rotation_penalty).replace(".", "")
                                + "su"
                                + str(score_upscale).replace(".", "")
                            )

                            results[name] = {
                                "window_influence": window_influence,
                                "score_upscale": score_upscale,
                                "rotation_penalty": rotation_penalty,
                                "rotation_step": rotation_step,
                                "rotations_count": rotations_count,
                            }
        return results

    eval_kwargs = create_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [3, 2]:
            for size in [1, -1]:
                for context_amount in [0.1, -0.1, -0.2, 0.2, 0]:

                    target_size = [127, 127] if size == 1 else [-1, -1]
                    search_size = [255, 255] if size == 1 else [-1, -1]

                    name = (
                        "1r-b"
                        + str(feature_blocks)
                        + ("-us" if size == 1 else "-os")
                        + "-c"
                        + str(context_amount).replace(".", "")
                    )
                    result.append(
                        (
                            Model(
                                name,
                                feature_blocks=feature_blocks,
                                target_size=target_size,
                                search_size=search_size,
                                context_amount=context_amount,
                                train_steps=20000,
                            ),
                            eval_kwargs,
                        )
                    )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best(device_id=0, total_devices=4):
    def create_eval_kwargs():
        params = {
            "window_influence": [0.15, 0.25, 0.35],
            "score_upscale": [16],
            "rotation_penalty": [0.98, 0.96],
            "rotation_step": [0.15, 0.1, 0.075],
            "rotations_count": [3, 5],
            "target_feature_merge_scale": [0, 0.1, 0.3, 0.5, 0.7],
        }

        results = {}

        for window_influence in params["window_influence"]:
            for score_upscale in params["score_upscale"]:
                for rotation_penalty in params["rotation_penalty"]:
                    for rotation_step in params["rotation_step"]:
                        for rotations_count in params["rotations_count"]:
                            for target_feature_merge_scale in params[
                                "target_feature_merge_scale"
                            ]:
                                name = (
                                    str(rotations_count).replace(".", "")
                                    + "r"
                                    + str(rotation_step).replace(".", "")
                                    + "-rp"
                                    + str(rotation_penalty).replace(".", "")
                                    + "su"
                                    + str(score_upscale).replace(".", "")
                                    + "tfms"
                                    + str(target_feature_merge_scale).replace(".", "")
                                )

                                results[name] = {
                                    "window_influence": window_influence,
                                    "score_upscale": score_upscale,
                                    "rotation_penalty": rotation_penalty,
                                    "rotation_step": rotation_step,
                                    "rotations_count": rotations_count,
                                    "target_feature_merge_scale": target_feature_merge_scale,
                                }
        return results

    eval_kwargs = create_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.5, 0.7]:

                    target_size = [127, 127] if size == 1 else [-1, -1]
                    search_size = [255, 255] if size == 1 else [-1, -1]

                    name = (
                        "2r-b"
                        + str(feature_blocks)
                        + ("-us" if size == 1 else "-os")
                        + "-c"
                        + str(context_amount).replace(".", "")
                    )
                    result.append(
                        (
                            Model(
                                name,
                                feature_blocks=feature_blocks,
                                target_size=target_size,
                                search_size=search_size,
                                context_amount=context_amount,
                                train_steps=8000,
                                save_step=200,
                                loads=[200, 600, 1200, 2000, 4000, 8000],
                            ),
                            eval_kwargs,
                        )
                    )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_small_lr(device_id=0, total_devices=4):
    def create_eval_kwargs():
        params = {
            "window_influence": [0.15, 0.25, 0.35],
            "score_upscale": [16],
            "rotation_penalty": [0.98, 0.96],
            "rotation_step": [0.15, 0.1, 0.075],
            "rotations_count": [3, 5],
            "target_feature_merge_scale": [0, 0.1, 0.3, 0.5, 0.7],
        }

        results = {}

        for window_influence in params["window_influence"]:
            for score_upscale in params["score_upscale"]:
                for rotation_penalty in params["rotation_penalty"]:
                    for rotation_step in params["rotation_step"]:
                        for rotations_count in params["rotations_count"]:
                            for target_feature_merge_scale in params[
                                "target_feature_merge_scale"
                            ]:
                                name = (
                                    str(rotations_count).replace(".", "")
                                    + "r"
                                    + str(rotation_step).replace(".", "")
                                    + "-rp"
                                    + str(rotation_penalty).replace(".", "")
                                    + "su"
                                    + str(score_upscale).replace(".", "")
                                    + "tfms"
                                    + str(target_feature_merge_scale).replace(".", "")
                                )

                                results[name] = {
                                    "window_influence": window_influence,
                                    "score_upscale": score_upscale,
                                    "rotation_penalty": rotation_penalty,
                                    "rotation_step": rotation_step,
                                    "rotations_count": rotations_count,
                                    "target_feature_merge_scale": target_feature_merge_scale,
                                }
        return results

    eval_kwargs = create_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.2, 0.5]:
                    for lr in [0.00001, 0.000005]:

                        target_size = [127, 127] if size == 1 else [-1, -1]
                        search_size = [255, 255] if size == 1 else [-1, -1]

                        name = (
                            "3rlr-b"
                            + str(feature_blocks)
                            + ("-us" if size == 1 else "-os")
                            + "-c"
                            + str(context_amount).replace(".", "")
                        )
                        result.append(
                            (
                                Model(
                                    name,
                                    feature_blocks=feature_blocks,
                                    target_size=target_size,
                                    search_size=search_size,
                                    context_amount=context_amount,
                                    train_steps=50000,
                                    save_step=1000,
                                    loads=[
                                        1000,
                                        2000,
                                        4000,
                                        8000,
                                        16000,
                                        32000,
                                        50000,
                                    ],
                                    lr=lr,
                                ),
                                eval_kwargs,
                            )
                        )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_small_lr_small_rpos(device_id=0, total_devices=4):
    def create_eval_kwargs():
        params = {
            "window_influence": [0.15, 0.25, 0.35],
            "score_upscale": [16],
            "rotation_penalty": [0.98, 0.96],
            "rotation_step": [0.15, 0.1, 0.075],
            "rotations_count": [3, 5],
            "target_feature_merge_scale": [0, 0.1, 0.3, 0.5, 0.7],
        }

        results = {}

        for window_influence in params["window_influence"]:
            for score_upscale in params["score_upscale"]:
                for rotation_penalty in params["rotation_penalty"]:
                    for rotation_step in params["rotation_step"]:
                        for rotations_count in params["rotations_count"]:
                            for target_feature_merge_scale in params[
                                "target_feature_merge_scale"
                            ]:
                                name = (
                                    str(rotations_count).replace(".", "")
                                    + "r"
                                    + str(rotation_step).replace(".", "")
                                    + "-rp"
                                    + str(rotation_penalty).replace(".", "")
                                    + "su"
                                    + str(score_upscale).replace(".", "")
                                    + "tfms"
                                    + str(target_feature_merge_scale).replace(".", "")
                                )

                                results[name] = {
                                    "window_influence": window_influence,
                                    "score_upscale": score_upscale,
                                    "rotation_penalty": rotation_penalty,
                                    "rotation_step": rotation_step,
                                    "rotations_count": rotations_count,
                                    "target_feature_merge_scale": target_feature_merge_scale,
                                }
        return results

    eval_kwargs = create_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.2, 0.5]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 2, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "6rlr-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=64000,
                                        save_step=2000,
                                        loads=[2000, 8000, 16000, 32000, 64000,],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_small_lr_small_rpos_2(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.2, 0.5]:
                    for lr in [0.000002, 0.00001]:
                        for r_pos in [4, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "6rlr2-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=256000,
                                        save_step=2000,
                                        loads=[2000, 32000, 64000, 128000, 256000,],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_focall_loss(device_id=0, total_devices=4):
    def create_eval_kwargs():
        params = {
            "window_influence": [0.15, 0.25, 0.35],
            "score_upscale": [16],
            "rotation_penalty": [0.98, 0.96],
            "rotation_step": [0.15, 0.1, 0.075],
            "rotations_count": [3, 5],
            "target_feature_merge_scale": [0, 0.1, 0.3, 0.5, 0.7],
        }

        results = {}

        for window_influence in params["window_influence"]:
            for score_upscale in params["score_upscale"]:
                for rotation_penalty in params["rotation_penalty"]:
                    for rotation_step in params["rotation_step"]:
                        for rotations_count in params["rotations_count"]:
                            for target_feature_merge_scale in params[
                                "target_feature_merge_scale"
                            ]:
                                name = (
                                    str(rotations_count).replace(".", "")
                                    + "r"
                                    + str(rotation_step).replace(".", "")
                                    + "-rp"
                                    + str(rotation_penalty).replace(".", "")
                                    + "su"
                                    + str(score_upscale).replace(".", "")
                                )

                                results[name] = {
                                    "window_influence": window_influence,
                                    "score_upscale": score_upscale,
                                    "rotation_penalty": rotation_penalty,
                                    "rotation_step": rotation_step,
                                    "rotations_count": rotations_count,
                                    "target_feature_merge_scale": target_feature_merge_scale,
                                }
        return results

    eval_kwargs = create_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [3, 1, 2]:
            for size in [-1]:
                for context_amount in [0.2, 0.5]:
                    for loss_function in ["focal"]:

                        target_size = [127, 127] if size == 1 else [-1, -1]
                        search_size = [255, 255] if size == 1 else [-1, -1]

                        name = (
                            "4fl-b"
                            + str(feature_blocks)
                            + ("-us" if size == 1 else "-os")
                            + "-c"
                            + str(context_amount).replace(".", "")
                            + "-lf"
                            + str(loss_function).replace(".", "")
                        )
                        result.append(
                            (
                                Model(
                                    name,
                                    feature_blocks=feature_blocks,
                                    target_size=target_size,
                                    search_size=search_size,
                                    context_amount=context_amount,
                                    train_steps=50000,
                                    save_step=1000,
                                    loads=[
                                        1000,
                                        2000,
                                        4000,
                                        8000,
                                        16000,
                                        32000,
                                        50000,
                                    ],
                                    loss_function=loss_function,
                                ),
                                eval_kwargs,
                            )
                        )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def create_selected_eval_kwargs():
    params = {
        "window_influence": [0.35],
        "score_upscale": [16],
        "rotation_penalty": [0.98],
        "rotation_step": [0.15, 0.1],
        "rotations_count": [3],
        "target_feature_merge_scale": [0, 0.01],
    }

    results = {}

    for window_influence in params["window_influence"]:
        for score_upscale in params["score_upscale"]:
            for rotation_penalty in params["rotation_penalty"]:
                for rotation_step in params["rotation_step"]:
                    for rotations_count in params["rotations_count"]:
                        for target_feature_merge_scale in params[
                            "target_feature_merge_scale"
                        ]:
                            name = (
                                str(rotations_count).replace(".", "")
                                + "r"
                                + str(rotation_step).replace(".", "")
                                + "-rp"
                                + str(rotation_penalty).replace(".", "")
                                + "su"
                                + str(score_upscale).replace(".", "")
                                + "tfms"
                                + str(target_feature_merge_scale).replace(".", "")
                            )

                            results[name] = {
                                "window_influence": window_influence,
                                "score_upscale": score_upscale,
                                "rotation_penalty": rotation_penalty,
                                "rotation_step": rotation_step,
                                "rotations_count": rotations_count,
                                "target_feature_merge_scale": target_feature_merge_scale,
                            }
    return results


def create_extended_eval_kwargs():
    params = {
        "window_influence": [0.35, 0.45],
        "score_upscale": [8, 16],
        "rotation_penalty": [0.98, 0.96],
        "rotation_step": [0.15, 0.1, 0.075, 0.04],
        "rotations_count": [3, 5],
        "target_feature_merge_scale": [0, 0.005, 0.01],
    }
    results = {}

    for window_influence in params["window_influence"]:
        for score_upscale in params["score_upscale"]:
            for rotation_penalty in params["rotation_penalty"]:
                for rotation_step in params["rotation_step"]:
                    for rotations_count in params["rotations_count"]:
                        for target_feature_merge_scale in params[
                            "target_feature_merge_scale"
                        ]:
                            name = (
                                str(rotations_count).replace(".", "")
                                + "r"
                                + str(rotation_step).replace(".", "")
                                + "-rp"
                                + str(rotation_penalty).replace(".", "")
                                + "su"
                                + str(score_upscale).replace(".", "")
                                + "wi"
                                + str(window_influence).replace(".", "")
                                + "tfms"
                                + str(target_feature_merge_scale).replace(".", "")
                            )

                            results[name] = {
                                "window_influence": window_influence,
                                "score_upscale": score_upscale,
                                "rotation_penalty": rotation_penalty,
                                "rotation_step": rotation_step,
                                "rotations_count": rotations_count,
                                "target_feature_merge_scale": target_feature_merge_scale,
                            }
    return results


def run_new(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks, backbone in [
            (1, "pp"),
            (1, "tanet"),
            (3, "spp"),
        ]:  # , (3, "stanet")
            for size in [-1]:
                for context_amount in [0.2, 0.5]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 2, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "n0-b"
                                + str(feature_blocks)
                                + "-"
                                + str(backbone).replace(".", "")
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        backbone=backbone,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=64000,
                                        save_step=2000,
                                        loads=[2000, 8000, 32000, 64000],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_new_smaller(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks, backbone in [
            (1, "pp"),
            (3, "spps"),
            # (3, "stanets"), worse results
            (3, "spp"),
            # (3, "stanet"), worse results
        ]:
            for size in [-1]:
                for context_amount in [0.2, -0.2]:
                    for lr in [0.0001, 0.000002]:
                        for r_pos in [4, 2, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "n3-b"
                                + str(feature_blocks)
                                + "-"
                                + str(backbone).replace(".", "")
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        backbone=backbone,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=128000,
                                        save_step=2000,
                                        loads=[2000, 32000, 64000, 128000],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_again(device_id=0, total_devices=4):

    eval_kwargs = create_extended_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.1, 0.3]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "s0-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=256000,
                                        save_step=2000,
                                        loads=[
                                            2000,
                                            8000,
                                            16000,
                                            32000,
                                            64000,
                                            72000,
                                            128000,
                                            168000,
                                            256000,
                                        ],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_ar(device_id=0, total_devices=4):

    eval_kwargs = create_extended_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.3, 0.2]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "ar1-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=1000,
                                        loads=[1000, 2000, 8000, 16000, 32000],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_ar_x(device_id=0, total_devices=4):

    eval_kwargs = create_extended_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.3, 0.2]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [6, 8]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "ar1-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=1000,
                                        loads=[1000, 2000, 8000, 16000, 32000,],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_sr(device_id=0, total_devices=4):

    eval_kwargs = create_extended_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1]:
            for size in [-1]:
                for context_amount in [0.2]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "sr1-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=1000,
                                        loads=[1000, 2000, 8000, 16000, 32000,],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_sr_mt(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1, 3]:
            for size in [-1]:
                for context_amount in [0.2, 0.5]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "srmt1-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=128000,
                                        save_step=1000,
                                        loads=[128000, 64000, 8000, 16000, 32000],
                                        lr=lr,
                                        r_pos=r_pos,
                                        track_ids=[
                                            "0005",
                                            "0006",
                                            "0007",
                                            "0008",
                                            "0009",
                                            "0010",
                                            "0011",
                                            "0012",
                                            "0013",
                                            "0014",
                                            "0015",
                                            "0016",
                                            "0017",
                                            "0018",
                                        ],
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_sr_mt1(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1, 3]:
            for size in [-1]:
                for context_amount in [0.2, 0.5]:
                    for lr in [0.0001]:
                        for r_pos in [16, 8]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "srmt1-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=128000,
                                        save_step=1000,
                                        loads=[128000, 64000, 8000, 16000, 32000],
                                        lr=lr,
                                        r_pos=r_pos,
                                        track_ids=[
                                            "0005",
                                            "0006",
                                            "0007",
                                            "0008",
                                            "0009",
                                            "0010",
                                            "0011",
                                            "0012",
                                            "0013",
                                            "0014",
                                            "0015",
                                            "0016",
                                            "0017",
                                            "0018",
                                        ],
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_sr_b3(device_id=0, total_devices=4):

    eval_kwargs = create_extended_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [3]:
            for size in [-1]:
                for context_amount in [0.2]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [8, 4]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "sr1-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=1000,
                                        loads=[1000, 2000, 8000, 16000, 32000,],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_best_sr_b3(device_id=0, total_devices=4):

    eval_kwargs = create_extended_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [3]:
            for size in [-1]:
                for context_amount in [0.2]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [8, 4]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "sr1-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=1000,
                                        loads=[1000, 2000, 8000, 16000, 32000,],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_new_smaller_b1(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks, backbone in [
            (1, "pp"),
            (1, "tanet"),
        ]:
            for size in [-1]:
                for context_amount in [0.2, 0.1]:
                    for lr in [0.0001, 0.000002]:
                        for r_pos in [4, 2, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "n4-b"
                                + str(feature_blocks)
                                + "-"
                                + str(backbone).replace(".", "")
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        backbone=backbone,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=64000,
                                        save_step=2000,
                                        loads=[2000, 8000, 16000, 32000, 64000],
                                        lr=lr,
                                        r_pos=r_pos,
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_search_small(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1, 3]:
            for size in [-1]:
                for context_amount in [0.1, 0.3]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "z0-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=2000,
                                        loads=[
                                            2000,
                                            8000,
                                            16000,
                                            32000,
                                        ],
                                        lr=lr,
                                        r_pos=r_pos,
                                        search_type="small"
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_search_small_1(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1, 3]:
            for size in [-1]:
                for context_amount in [0.1, 0.3]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [8, 16]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "z0-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=2000,
                                        loads=[
                                            2000,
                                            8000,
                                            16000,
                                            32000,
                                        ],
                                        lr=lr,
                                        r_pos=r_pos,
                                        search_type="small"
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_search_small_2(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1, 3]:
            for size in [-1]:
                for context_amount in [0.2, 0.4]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [8, 16]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "z0-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=2000,
                                        loads=[
                                            2000,
                                            8000,
                                            16000,
                                            32000,
                                        ],
                                        lr=lr,
                                        r_pos=r_pos,
                                        search_type="small"
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)


def run_search_small_3(device_id=0, total_devices=4):

    eval_kwargs = create_selected_eval_kwargs()

    def create_models(eval_kwargs):
        result = []
        for feature_blocks in [1, 3]:
            for size in [-1]:
                for context_amount in [0.2, 0.4]:
                    for lr in [0.00001, 0.000002]:
                        for r_pos in [4, 1]:
                            target_size = [127, 127] if size == 1 else [-1, -1]
                            search_size = [255, 255] if size == 1 else [-1, -1]

                            name = (
                                "z0-b"
                                + str(feature_blocks)
                                + ("-us" if size == 1 else "-os")
                                + "-c"
                                + str(context_amount).replace(".", "")
                                + "-lr"
                                + str(lr).replace(".", "")
                                + "-rpos"
                                + str(r_pos).replace(".", "")
                            )
                            result.append(
                                (
                                    Model(
                                        name,
                                        feature_blocks=feature_blocks,
                                        target_size=target_size,
                                        search_size=search_size,
                                        context_amount=context_amount,
                                        train_steps=32000,
                                        save_step=2000,
                                        loads=[
                                            2000,
                                            8000,
                                            16000,
                                            32000,
                                        ],
                                        lr=lr,
                                        r_pos=r_pos,
                                        search_type="small"
                                    ),
                                    eval_kwargs,
                                )
                            )

        return result

    models = create_models(eval_kwargs)

    i = device_id

    while i < len(models):
        model, eval_kwargs = models[i]
        i += total_devices

        result = model.eval_and_train(
            device="cuda:" + str(device_id), eval_kwargs=eval_kwargs
        )
        print(result)



if __name__ == "__main__":

    fire.Fire()

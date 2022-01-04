import sys
import os
import torch
import fire
from multiprocessing import Process, set_start_method, get_context

from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.test import (
    test_pp_siamese_fit,
    test_rotated_pp_siamese_eval,
    test_rotated_pp_siamese_infer,
)

default_track_ids = [
    "0000",
    "0001",
    "0002",
    "0003",
    "0004",
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
    # "0017",
    # "0018",
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
        **kwargs
    ) -> None:
        self.model_name = model_name
        self.train_steps = train_steps
        self.save_step = save_step
        self.loads = loads
        self.track_ids = track_ids
        self.iou_min = iou_min
        self.kwargs = kwargs
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

        last_checkpoint_path = "./temp/" + self.model_name + "/checkpoints" + f"/checkpoint_{steps}.pth"

        if os.path.exists(last_checkpoint_path):
            print("The model is already trained")
        else:
            test_pp_siamese_fit(
                self.model_name,
                0,
                steps,
                False,
                checkpoint_after_iter=self.save_step,
                device=device,
                **self.kwargs
            )

        if eval_after:
            return self.eval(
                steps,
                self.loads,
                device,
                self.track_ids,
                eval_kwargs=eval_kwargs,
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
                    **kwargs
                )
                results[str(load) + "_" + str(id)] = result

        return results

    def eval_and_train(self, device, eval_kwargs={"default": {}}):
        return self.train(
            self.train_steps, device, True, eval_kwargs=eval_kwargs
        )


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
                    float(values["total_mean_precision"] if "total_mean_precision" in values else -1),
                    float(values["total_mean_success"] if "total_mean_success" in values else -1),
                ]
                results.append(result)

    results = sorted(results, key=lambda x: x[2])
    for name, iou3d, precision, success in results:
        print(name, "precision", precision, "success", success, "iou3d", iou3d)


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



if __name__ == "__main__":

    fire.Fire()

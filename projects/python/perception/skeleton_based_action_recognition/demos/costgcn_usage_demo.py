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

import argparse
import torch

from opendr.perception.skeleton_based_action_recognition.continual_stgcn_learner import (
    CoSTGCNLearner, _MODEL_NAMES
)
from opendr.engine.datasets import ExternalDataset
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", help="Fit the model", default=False, action="store_true")
    parser.add_argument("--num_fit_steps", help="Numer of steps to fit the model", type=int, default=10)
    parser.add_argument("--eval", help="Evaluate the model", default=False, action="store_true")
    parser.add_argument("--optimize", help="Perform inference using the model", default=False, action="store_true")
    parser.add_argument("--infer", help="Perform inference using the model", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cpu")
    parser.add_argument("--backbone", help="The model type to use", type=str, default="costgcn", choices=_MODEL_NAMES)
    parser.add_argument("--batch_size", help="The batch size of the model", type=int, default=2)
    return parser.parse_args()


def main(args):
    tmp_path = Path(__file__).parent / "tmp"

    # Define learner
    learner = CoSTGCNLearner(
        device=args.device,
        temp_path=str(tmp_path),
        batch_size=args.batch_size,
        backbone=args.backbone,
        num_workers=0,
    )

    pretrained_weights_path = learner.download(
        path=str(tmp_path / "pretrained_models"),
        method_name=args.backbone,
        mode="pretrained",
        file_name=f"{args.backbone}_ntu60_xview_joint.ckpt",
    )

    learner.load(pretrained_weights_path)

    # Define datasets
    data_path = tmp_path / "data"

    train_ds_path = learner.download(mode="train_data", path=str(data_path))
    val_ds_path = learner.download(mode="val_data", path=str(data_path))

    train_ds = learner._prepare_dataset(
        ExternalDataset(path=train_ds_path, dataset_type="NTURGBD"),
        data_filename="train_joints.npy",
        labels_filename="train_labels.pkl",
        skeleton_data_type="joint",
        phase="train",
        verbose=False,
    )

    val_ds = learner._prepare_dataset(
        ExternalDataset(path=val_ds_path, dataset_type="NTURGBD"),
        data_filename="val_joints.npy",
        labels_filename="val_labels.pkl",
        skeleton_data_type="joint",
        phase="val",
        verbose=False,
    )

    # Invoke operations
    if args.fit:
        learner.fit(dataset=train_ds, val_dataset=val_ds, steps=args.num_fit_steps)

    if args.eval:
        results = learner.eval(val_ds)
        print("Evaluation results: ", results)

    if args.optimize:
        learner.optimize()

    if args.infer:
        dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
        batch = next(iter(dl))[0]
        frame = batch[:, :, -1]  # Select a single frame

        categories = learner.infer(frame)
        print("Inferred :")
        for i, c in enumerate(categories):
            print(f"[{i}] category.data = {c.data}, category.confidence = {c.confidence.detach().numpy()}")


if __name__ == "__main__":
    main(parse_args())

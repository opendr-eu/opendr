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

from opendr.perception.activity_recognition import CoTransEncLearner
from opendr.perception.activity_recognition.datasets import DummyTimeseriesDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", help="Fit the model", default=False, action="store_true")
    parser.add_argument("--num_fit_steps", help="Numer of steps to fit the model", type=int, default=10)
    parser.add_argument("--eval", help="Evaluate the model", default=False, action="store_true")
    parser.add_argument("--optimize", help="Perform inference using the model", default=False, action="store_true")
    parser.add_argument("--infer", help="Perform inference using the model", default=False, action="store_true")
    parser.add_argument("--device", help="Device to use (cpu, cuda)", type=str, default="cpu")
    parser.add_argument("--input_dims", help="Input dimensionality of the model and dataset", type=float, default=8)
    parser.add_argument("--hidden_dims", help="The number of hidden dimensions of the model", type=float, default=32)
    parser.add_argument("--sequence_len", help="The length of the time-series to consider", type=int, default=64)
    parser.add_argument("--num_heads", help="Number of attention heads to employ", type=int, default=8)
    parser.add_argument("--batch_size", help="The batch size of the model", type=int, default=2)
    return parser.parse_args()


def main(args):
    # Define learner
    learner = CoTransEncLearner(
        batch_size=args.batch_size,
        device="cpu",
        input_dims=args.input_dims,
        hidden_dims=args.hidden_dims,
        sequence_len=args.sequence_len,
        num_heads=args.num_heads,
        num_classes=4,
    )

    # Define datasets
    train_ds = DummyTimeseriesDataset(
        sequence_len=args.sequence_len,
        num_sines=args.input_dims,
        num_datapoints=args.sequence_len * 2,
    )
    val_ds = DummyTimeseriesDataset(
        sequence_len=args.sequence_len,
        num_sines=args.input_dims,
        num_datapoints=args.sequence_len * 2,
        base_offset=args.sequence_len * 2,
    )
    test_ds = DummyTimeseriesDataset(
        sequence_len=args.sequence_len,
        num_sines=args.input_dims,
        num_datapoints=args.sequence_len * 2,
        base_offset=args.sequence_len * 4,
    )

    # Invoke operations
    if args.fit:
        learner.fit(dataset=train_ds, val_dataset=val_ds, steps=args.num_fit_steps)

    if args.eval:
        results = learner.eval(test_ds)
        print("Evaluation results: ", results)

    if args.optimize:
        learner.optimize()

    if args.infer:
        dl = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
        tensor = next(iter(dl))[0][0]
        category = learner.infer(tensor)
        print(f"Inferred category.data = {category.data}, category.confidence = {category.confidence.detach().numpy()}")


if __name__ == "__main__":
    main(parse_args())

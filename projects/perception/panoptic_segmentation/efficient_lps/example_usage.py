#!/usr/bin/env python
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

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from opendr.engine.data import PointCloud
from opendr.perception.panoptic_segmentation import EfficientLpsLearner, SemanticKittiDataset, EfficientPsLearner


def download_models(*, cp_dir, **_):
	"""
	TODO:
	:param cp_dir:
	:return:
	"""

	EfficientLpsLearner.download(cp_dir, trained_on='nuscenes')
	EfficientLpsLearner.download(cp_dir, trained_on='kitti')


def train(*, kitti_dir, cp_dir, log_dir, **_):
	"""
	TODO:

	:param kitti_dir:
	:type kitti_dir:
	:param cp_dir:
	:param log_dir:
	:return:
	"""

	train_data = SemanticKittiDataset(kitti_dir, split="train")
	val_data = SemanticKittiDataset(kitti_dir, split="valid")

	learner = EfficientLpsLearner(
		iters=2,
		batch_size=1,
		checkpoint_after_iter=2
	)

	train_stats = learner.fit(train_data, val_dataset=val_data,
				logging_path=str(log_dir))

	learner.save(path=cp_dir)

	assert train_stats  # This assert is just a workaround since pyflakes does not support the NOQA comment


def evaluate(*, kitti_dir, cp_dir, **_):
	"""
	TODO:
	:param kitti_dir:
	:param cp_dir:
	:return:
	"""

	val_dataset = SemanticKittiDataset(path=kitti_dir, split="valid")

	learner = EfficientLpsLearner()
	learner.load(path=f'{cp_dir}/model.pth')
	eval_stats = learner.eval(val_dataset, print_results=True)
	assert eval_stats  # This assert is just a workaround since pyflakes does not support the NOQA comment


def inference(*, kitti_dir, cp_dir, projected=False,
			  save_figure=False, display_figure=False,
			  detailed=False, **_):
	"""
	TODO
	:param kitti_dir:
	:param cp_dir:
	:param projected:
	:param save_figure:
	:param display_figure:
	:param detailed:
	:return:
	"""

	pointcloud_filenames = [
		f'{kitti_dir}/sequences/00/velodyne/002250.bin',
		f'{kitti_dir}/sequences/08/velodyne/002000.bin',
		f'{kitti_dir}/sequences/15/velodyne/000950.bin',
	]
	clouds = [PointCloud(np.fromfile(f, dtype=np.float32).reshape(-1, 4)) for f in pointcloud_filenames]

	learner = EfficientLpsLearner()
	learner.load(path=f'{cp_dir}/model.pth')
	predictions = learner.infer(clouds, projected=projected)
	for cloud, prediction, f in zip(clouds, predictions, pointcloud_filenames):
		filename = Path(f).with_suffix(".png").name
		# Clip values since the Cityscapes palette only has 19 colors, plus black added in the visualize method
		semantics = prediction[1]
		if projected:
			semantics = semantics.data
		semantics[semantics > 18] = 19
		if projected:
			EfficientPsLearner.visualize(prediction[-1], prediction[:2],
										 show_figure=display_figure,
										 save_figure=save_figure, figure_filename=filename,
										 detailed=detailed)
		else:
			EfficientLpsLearner.visualize(cloud, prediction[:2],
										  show_figure=display_figure,
										  save_figure=save_figure, figure_filename=filename,
										  detailed=detailed)


def parse_args() -> dict:
	"""
	TODO
	:return:
	"""

	# Default values
	dft_kitti_dir = "~/dat/kitti/dataset/"
	dft_cp_dir = "~/dat/cp/efficient_lps/"
	dft_log_dir = "~/dat/log/"

	def _resolve_path(path: str) -> str:
		"""
		TODO
		:param path:
		:return:
		"""
		return str(Path(path).expanduser().resolve())

	parser = ArgumentParser()

	parser.add_argument("--kitti_dir", "-k", type=_resolve_path, default=dft_kitti_dir,
						help="Directory where the KITTY dataset is located, parent of the sequences/ folder.")
	parser.add_argument("--cp_dir", "-c", type=_resolve_path, default=dft_cp_dir,
						help="Directory where the model checkpoints are to be saved.")
	parser.add_argument("--log_dir", "-l", type=_resolve_path, default=dft_log_dir,
						help="Directory where the logs are to be saved.")

	parser.add_argument("--skip_download", action="store_true",
						help="Skip the Download test.")
	parser.add_argument("--skip_train", action="store_true",
						help="Skip the Training test.")
	parser.add_argument("--skip_eval", action="store_true",
						help="Skip the Evaluation test.")
	parser.add_argument("--skip_infer", action="store_true",
						help="Skip the Inference test.")

	parser.add_argument("--display_figure", "-v", action="store_true",
						help="Display the inferred data figures.")
	parser.add_argument("--save_figure", "-s", action="store_true",
						help="Save the inferred data figures.")
	parser.add_argument("--projected", "-p", action="store_true",
						help="Inferred data is returned as spherical projection.")
	parser.add_argument("--detailed", "-d", action="store_true",
						help="Inferred image is shown in detail (Image, Panoptic, Semantic, Contours).")

	kwargs = vars(parser.parse_args())

	return kwargs


def main():
	kwargs = parse_args()

	if not kwargs["skip_download"]:
		print("-" * 40 + "\n===> Downloading Pretrained Models:")
		download_models(**kwargs)
		print("Download succeeded\n" + "-" * 40 + "\n")

	if not kwargs["skip_train"]:
		print("-" * 40 + "\n===> Training Model:")
		train(**kwargs)
		print("Training succeeded\n" + "-" * 40 + "\n")

	if not kwargs["skip_eval"]:
		print("-" * 40 + "\n===> Evaluating Model:")
		evaluate(**kwargs)
		print("Evaluation succeeded\n" + "-" * 40 + "\n")

	if not kwargs["skip_infer"]:
		print("-" * 40 + "\n===> Inference with Pre-trained")
		inference(**kwargs)
		print("Inference succeeded\n" + "-" * 40 + "\n")


if __name__ == "__main__":
	main()

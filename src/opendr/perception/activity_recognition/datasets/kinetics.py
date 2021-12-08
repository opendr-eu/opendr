# Copyright 2020-2021 OpenDR Project
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

import av
import json
import random
import torch
import zipfile
import os
from opendr.engine.datasets import ExternalDataset, DatasetIterator
from pathlib import Path
from logging import getLogger
from tqdm.contrib.concurrent import process_map
from functools import partial
from typing import List, Optional, Tuple, Union
from joblib import Memory
from opendr.perception.activity_recognition.datasets.utils import decoder
from opendr.perception.activity_recognition.datasets.utils.transforms import standard_video_transforms
from opendr.engine.constants import OPENDR_SERVER_URL
from urllib.request import urlretrieve
import pandas as pd

logger = getLogger(__file__)

CLASSES = pd.read_csv(
    Path(__file__).parent / "kinetics400_classes.csv", verbose=True, index_col=0
).to_dict()["name"]


class KineticsDataset(ExternalDataset, DatasetIterator, torch.utils.data.Dataset):
    """
    Kinetics dataset for Trimmed Activity Recognition

    It has multiple versions available which are identified by their number of classes (400, 600, 700)

    Splits can be downloaded from:
    https://deepmind.com/research/open-source/open-source-datasets/kinetics/

    Videos must be downloaded using a YouTube crawler, e.g.:
    https://github.com/LukasHedegaard/youtube-dataset-downloader
    """

    def __init__(
        self,
        path: str,
        frames_per_clip: int,
        step_between_clips=1,
        temporal_downsampling=1,
        split="train",
        video_transform=None,
        use_caching=False
    ):
        """
        Kinetics dataset

        This dataset consider every video as a collection of video clips of fixed size, specified
        by ``frames_per_clip``, where the step in frames between each clip is given by
        ``step_between_clips``.

        Args:
            path (str): path directory of the Kinetics dataset folder, containing subfolders "data" and "splits"
            frames_per_clip (int): number of frames in a clip
            step_between_clips (int): number of frames between each clip. Defaults to 1.
            temporal_downsampling (int): rate of downsampling in time. Defaults to 1.
            split (str, optional): Which split to use (Options are ["train", "val", "test"]). Defaults to "train".
            video_transform (callable, optional): A function/transform that takes in a TxHxWxC video
                and returns a transformed version. If None, a standard video transform will be applied. Defaults to None.
            use_caching (bool): Cache long-running operations. Defaults to False.

        """
        ExternalDataset.__init__(self, path=str(path), dataset_type="kinetics")
        DatasetIterator.__init__(self)
        torch.utils.data.Dataset.__init__(self)

        self.root_path = Path(path) / "data"
        self.annotation_path = Path(path) / "splits"
        self.num_retries = 10
        assert split in [
            "train",
            "val",
            "validate",
            "test",
        ], "Split '{}' not supported for Kinetics".format(split)
        if split == "val":
            split = "validate"
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.target_fps = 30
        self.temporal_downsampling = temporal_downsampling

        if video_transform:
            self.video_transform = video_transform
        else:
            train_transform, eval_transform = standard_video_transforms()
            self.video_transform = train_transform if self.split == "train" else eval_transform

        validate_splits = Memory(Path(os.getcwd()) / ".cache", verbose=1).cache(
            _validate_splits
        ) if use_caching else _validate_splits

        (
            self.labels,
            self.file_paths,
            self.spatial_temporal_idx,
            self.video_meta,
            self.classes,
            _,  # num_not_found,
            self.video_inds,
        ) = validate_splits(
            self.root_path,
            self.annotation_path,
            split
        )
        if len(self.classes) not in {400, 600, 700}:
            logger.warning(
                f"Only found {len(self.classes)} classes for {split} set, but expected either 400, 600, or 700 for Kinetics."
            )

    def __getitem__(self, idx):
        """
        This method is used for loading the idx-th sample of a dataset along with its annotation.

        :param idx: the index of the sample to load
        :return: the idx-th sample and its annotation
        :rtype: Tuple of (Data, Target)

        Returns tuple of:
            video (Tensor[T, H, W, C]): the `T` video frames
            label (int): class of the video clip
        """
        video_container = None
        frames = None
        for _ in range(self.num_retries):
            try:
                video_container = _get_video_container(
                    self.file_paths[idx], multi_thread_decode=False, backend="pyav",
                )
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self.file_paths[idx], e
                    )
                )
            # Select a random video if the current video was not able to access.
            if video_container is None:
                idx = random.randint(0, len(self.file_paths) - 1)
                continue

            # Decode video. Meta info is used to perform selective decoding.
            temporal_clip_idx = {
                "train": -1,  # pick random
                "validate": 0,  # pick middle of clip
                "test": 0,  # pick middle of clip
            }[self.split]
            frames = decoder.decode(
                video_container,
                self.temporal_downsampling,
                self.frames_per_clip,
                clip_idx=temporal_clip_idx,
                num_clips=1,
                video_meta=self.video_meta[idx],
                target_fps=self.target_fps,
                backend="pyav",
            )

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                idx = random.randint(0, len(self.file_paths) - 1)
            else:
                break

        if frames is None:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(self.num_retries)
            )

        video = frames
        label = self.labels[idx]

        if self.video_transform is not None:
            video = self.video_transform(video)

        sample = (video, label)

        return sample

    def __len__(self):
        """
        This method returns the size of the dataset.

        :return: the size of the dataset
        :rtype: int
        """
        return len(self.file_paths)

    @staticmethod
    def download_mini(path: Union[str, Path]):
        """Download mini version of dataset: One video of each class in Kinetics400

        Args:
            path (Union[str, Path]): Directory in which to store dataset
        """
        path = Path(path)
        if path.exists():
            logger.info("Kinetics400 mini already exists. Skipping download.")
            return

        path.mkdir(parents=True, exist_ok=True)

        url = os.path.join(
            OPENDR_SERVER_URL,
            "perception",
            "activity_recognition",
            "datasets",
            "kinetics400mini.zip"
        )
        zip_path = str(Path(path) / "kinetics400mini.zip")
        unzip_path = str(Path(path))

        logger.info(f"Downloading Kinetics400 mini from {url}")
        urlretrieve(url=url, filename=zip_path)

        logger.info(f"Unzipping Kinetics400 mini to {(unzip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        os.remove(zip_path)

    @staticmethod
    def download_micro(path: Union[str, Path]):
        """Download micro version of dataset: One video of first three classes in Kinetics400

        Args:
            path (Union[str, Path]): Directory in which to store dataset
        """
        path = Path(path)
        if path.exists():
            logger.info("Kinetics3 already exists. Skipping download.")
            return

        path.mkdir(parents=True, exist_ok=True)

        url = os.path.join(
            OPENDR_SERVER_URL,
            "perception",
            "activity_recognition",
            "datasets",
            "kinetics3.zip"
        )
        zip_path = str(Path(path) / "kinetics3.zip")
        unzip_path = str(Path(path))

        logger.info(f"Downloading Kinetics3 from {url}")
        urlretrieve(url=url, filename=zip_path)

        logger.info(f"Unzipping Kinetics3 to {(unzip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        os.remove(zip_path)


def _make_path_name(
    key_and_annotations: Tuple[str, dict], root_path: Path, extention="mp4"
) -> Optional[Tuple[Path, Optional[str]]]:
    key, annotation = key_and_annotations

    def map_segments(segment_arr: List[float]) -> Tuple[str, str]:
        assert len(segment_arr) == 2
        return (
            str(int(segment_arr[0])).zfill(6),
            str(int(segment_arr[1])).zfill(6),
        )

    start, stop = map_segments(annotation["annotations"]["segment"])
    if "label" in annotation["annotations"] and annotation["annotations"]["label"]:
        label = annotation["annotations"]["label"]
        p = (
            root_path / annotation["subset"] / label / f"{key}_{start}_{stop}.{extention}"
        ).resolve()
    else:
        p = (
            root_path / annotation["subset"] / f"{key}_{start}_{stop}.{extention}"
        ).resolve()
        label = None

    return (p, label) if p.exists() else None


def _validate_splits(
    root: str,
    annotation_path: str,
    split: str,
):
    root_path = Path(root)
    assert root_path.is_dir()

    annotation_file = Path(annotation_path) / f"{split}.json"
    assert annotation_file.exists()

    with open(annotation_file, "r") as f:
        annotations = json.load(f)

    # Validate annotations file
    test_key = next(iter(annotations))
    test_value = annotations[test_key]
    assert type(test_value) == dict
    assert "subset" in test_value
    assert "annotations" in test_value
    assert "label" in test_value["annotations"]
    assert "segment" in test_value["annotations"]

    # Determine valid files
    maybe_results = process_map(
        partial(_make_path_name, root_path=root_path,),
        annotations.items(),
        chunksize=4,
        desc=f"Validating Kinetics {split}",
    )

    results = list(filter(bool, maybe_results))
    num_not_found = len(maybe_results) - len(results)
    logger.info(
        f"{len(results)} / {len(maybe_results)} ({100*len(results)/len(maybe_results):.1f}%) of videos were valid"
    )

    assert len(results) > 0
    file_paths, labels = zip(*results)
    file_paths, labels = list(file_paths), list(labels)

    classes = sorted(set(labels))  # type: ignore
    class_to_idx = {n: i for i, n in enumerate(classes)}
    labels = [class_to_idx[lbl] for lbl in labels]
    file_paths = [str(p) for p in file_paths]

    num_clips = 1

    _labels = []
    _file_paths = []
    _spatial_temporal_idx = []
    _video_meta = {}
    _video_idx = []
    for i in range(len(file_paths)):
        for j in range(num_clips):
            _labels.append(labels[i])
            _file_paths.append(file_paths[i])
            _spatial_temporal_idx.append(j)
            _video_meta[i * num_clips + j] = {}
            _video_idx.append(i)

    return (
        _labels,
        _file_paths,
        _spatial_temporal_idx,
        _video_meta,
        classes,
        num_not_found,
        _video_idx,
    )


def _get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        with open(path_to_vid, "rb") as fp:
            container = fp.read()
        return container
    elif backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    else:
        raise NotImplementedError(f"Unknown backend {backend}")

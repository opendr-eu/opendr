import os
import pickle
import random
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
import torch.nn.functional as F

from opendr.perception.continual_slam.algorithm.depth_pose_module.replay_buffer.encoder import FeatureEncoder
from opendr.perception.continual_slam.continual_slam_learner import ContinualSLAMLearner
from opendr.perception.continual_slam.configs.config_parser import ConfigParser


class ReplayBuffer(TorchDataset):
    def __init__(self,
                 buffer_size: int,
                 save_memory: bool,
                 device: Optional[torch.device] = None,
                 dataset_config_path: Optional[str] = None,
                 height : Optional[int] = None,
                 width : Optional[int] = None,
                 save_state_path: Optional[Path] = None,
                 load_state_path: Optional[Path] = None,
                 local_save_path: Optional[Path] = None,
                 num_workers: int = 1,
                 cosine_similarity_threshold: float = 0.9,
                 num_features: int = 576,
                 sample_size: int = 3,
                 ):
        """
        This class implements a replay buffer for the depth pose module.
        """

        super().__init__()
        self.buffer_size = buffer_size
        self.save_memory = save_memory
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.save_state_path = save_state_path
        self.load_state_path = load_state_path
        self.local_save_path = local_save_path
        if dataset_config_path:
            self.dataset_config = ConfigParser(dataset_config_path).dataset
        if not (dataset_config_path or (height and width)):
            raise ValueError("Either dataset_config or height and width must be specified.")
        if dataset_config_path:
            self.height = self.dataset_config.height
            self.width = self.dataset_config.width
        else:
            self.height = height
            self.width = width
        self.num_workers = num_workers
        self.cosine_similarity_threshold = cosine_similarity_threshold
        self.num_features = num_features
        self.sample_size = sample_size
        self.frame_ids = [-1, 0, 1]
        # Count how many images are in the buffer
        self.count = 0

        # Create encoder
        self.encoder = FeatureEncoder(device=self.device, num_features=num_features)

        if self.save_memory:
            # Create image buffer tensor with shape (buffer_size, 3, 3, height, width)
            self.image_buffer = torch.zeros((buffer_size, 3, 3, self.height, self.width), dtype=torch.float32, device=self.device)
            self.feature_vector_buffer = torch.zeros((buffer_size, num_features), dtype=torch.float32, device=self.device)
            # Create distance buffer tensor with shape (buffer_size, 3)
            self.distance_buffer = torch.zeros((buffer_size, 3), dtype=torch.float32, device=self.device)

        else:
            if self.local_save_path is None:
                raise ValueError("local_save_path must be specified if save_memory is False.")
            # Read from disk
            # Create image buffer that holds file paths to the images
            self.image_buffer = []
            # Create distance buffer that holds file paths to the distance maps
            self.distance_buffer = []

            # Have to hold the feature vectors in memory
            self.feature_vector_buffer = torch.zeros((buffer_size, num_features), dtype=torch.float32, device=self.device)

            # Create a dictionary that holds the buffer
            os.makedirs(self.local_save_path, exist_ok=True)
            # Create a dictionary that holds the image buffer
            os.makedirs(os.path.join(self.local_save_path, "images"), exist_ok=True)
            # Create a dictionary that holds the distance buffer
            os.makedirs(os.path.join(self.local_save_path, "distances"), exist_ok=True)

        self.buffer =  {"image_buffer": self.image_buffer,
                        "distance_buffer": self.distance_buffer,
                        "feature_vector_buffer": self.feature_vector_buffer,}

    def add(self, data: Dict[Any, Tensor]):
        """
        This method adds a new image to the replay buffer.
        """

        # Format the data
        data = ContinualSLAMLearner._input_formatter(data)
        if self.count < self.buffer_size:
            if self.save_memory:
                for i in range(len(self.frame_ids)):
                    self.image_buffer[self.count, i] = data[(self.frame_ids[i], 'image')]
                    self.distance_buffer[self.count, i] = data[(self.frame_ids[i], 'distance')]

                    # Get the feature vector of the new image (central image from the image triplet)
                new_image_feature = self.encoder(data[(0, 'image')])
                self.feature_vector_buffer[self.count] = new_image_feature
            else:
                # Read from disk
                # Save into the disk
                for i in range(len(self.frame_ids)):
                    # Save the image to disk in the following format: image_{count}_{frame_id}.png
                    image_path = os.path.join(self.local_save_path,
                                              "images",
                                              f"image_{self.count}_{self.frame_ids[i]}.png")
                    # Send image from torch to PIL
                    image = data[(self.frame_ids[i], 'image')].cpu().numpy()
                    image = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))
                    image.save(image_path)
                    self.image_buffer.append(image_path)

                    # Save the distance map to disk in the following format: distance_{count}_{frame_id}.npy
                    distance_path = os.path.join(self.local_save_path,
                                                 "distances",
                                                 f"distance_{self.count}_{self.frame_ids[i]}.npy")
                    distance = data[(self.frame_ids[i], 'distance')].cpu().numpy()
                    np.save(distance_path, distance)
                    self.distance_buffer.append(distance_path)

                # Get the feature vector of the new image (central image from the image triplet)
                new_image_feature = self.encoder(data[(0, 'image')])
                self.feature_vector_buffer[self.count] = new_image_feature

            self.count += 1
        else:
            # Now we compare the new image with the images in the buffer using the encoder
            # and we add the new image to the buffer if it is sufficiently different from the
            # images in the buffer using cosine similarity.
            
            # Get the feature vector of the new image (central image from the image triplet)
            new_image_feature = self.encoder(data[(0, 'image')])
            if self.compute_cosine_similarity(new_image_feature):
                # Throw away a random image from the buffer
                random_index = random.randint(0, self.buffer_size - 1)
                if self.save_memory:
                    for i in range(len(self.frame_ids)):
                        self.image_buffer[random_index, i] = data[(self.frame_ids[i], 'image')]
                        self.distance_buffer[random_index, i] = data[(self.frame_ids[i], 'distance')]

                    self.feature_vector_buffer[random_index] = new_image_feature
                else:
                    # Change the image and distance map on disk
                    for i in range(len(self.frame_ids)):
                        # Save the image to disk in the following format: image_{random_index}_{frame_id}.png
                        image_path = os.path.join(self.local_save_path,
                                                  "images",
                                                  f"image_{random_index}_{self.frame_ids[i]}.png")
                        image = data[(self.frame_ids[i], 'image')].cpu().numpy()
                        image = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))
                        image.save(image_path)
                        self.image_buffer[random_index] = image_path

                        # Save the distance map to disk in the following format: distance_{count}_{frame_id}.npy
                        distance_path = os.path.join(self.local_save_path,
                                                     "distances",
                                                     f"distance_{random_index}_{self.frame_ids[i]}.npy")
                        distance = data[(self.frame_ids[i], 'distance')].cpu().numpy()
                        np.save(distance_path, distance)
                        self.distance_buffer[random_index] = distance_path

                    self.feature_vector_buffer[random_index] = new_image_feature
        if self.count % 20 == 0 and self.count != self.buffer_size:
            print(f"Replay buffer size: {self.count}")

    def sample(self) -> List:
        """
        This method returns a random sample of images from the replay buffer.
        """
        if self.count < self.sample_size:
            raise ValueError("The replay buffer does not have enough images to sample from.")
        else:
            # Sample random indices from the buffer
            random_indices = random.sample(range(self.count), self.sample_size)
            if self.save_memory:
                # Get the images and distances from the buffer
                image_sample = self.image_buffer[random_indices]
                distance_sample = self.distance_buffer[random_indices]
                batch = []
                for i in range(self.sample_size):
                    single_sample = {}
                    for j in range(len(self.frame_ids)):
                        single_sample[(self.frame_ids[j], 'image')] = image_sample[i, j]
                        single_sample[(self.frame_ids[j], 'distance')] = distance_sample[i, j]
                    batch.append(single_sample)
            else:
                # Read from disk
                batch = []
                for i in range(self.sample_size):
                    single_sample = {}
                    for j in range(len(self.frame_ids)):
                        image_path = self.image_buffer[random_indices[i]]
                        image = Image.open(image_path)
                        image = np.array(image)
                        single_sample[(self.frame_ids[j], 'image')] = image

                        distance_path = self.distance_buffer[random_indices[i]]
                        distance = np.load(distance_path)
                        single_sample[(self.frame_ids[j], 'distance')] = distance

                    batch.append(single_sample)
        return batch

    def save_state(self):
        state = {
            'buffer': self.buffer,
            'buffer_size': self.buffer_size,
            'count': self.count,
            'save_memory': self.save_memory,
            'device': self.device,
            'save_state_path': self.save_state_path,
            'load_state_path': self.load_state_path,
            'local_save_path': self.local_save_path,
        }
        with open(self.save_state_path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self):
        with open(self.load_state_path, 'rb') as f:
            state = pickle.load(f)
        self.buffer = state['buffer']
        self.image_buffer = self.buffer['image']
        self.distance_buffer = self.buffer['distance']
        self.feature_vector_buffer = self.buffer['feature_vector']
        self.buffer_size = state['buffer_size']
        self.save_memory = state['save_memory']
        self.device = state['device']
        self.save_state_path = state['save_state_path']
        self.load_state_path = state['load_state_path']
        self.local_save_path = state['local_save_path']

    def compute_cosine_similarity(self, feature_vector: Tensor) -> bool:
        """
        This method computes the cosine similarity between a feature vector and the feature vectors
        in the replay buffer.
        """
        # Compute the cosine similarity between the feature vector and the feature vectors in the buffer
        similarity = F.cosine_similarity(feature_vector, self.feature_vector_buffer)
        return similarity.max() < self.cosine_similarity_threshold

    def _recover_state(self):
        if self.load_state_path is not None:
            self.load_state()
        else:
            print('No state to load. Initializing new replay buffer.')

    def __getitem__(self) -> Dict[Any, Tensor]:
        return self.sample()

    def __len__(self):
        return self.count

import pickle
import random
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



class ReplayBuffer(TorchDataset):
    def __init__(self,
                 buffer_size: int,
                 save_memory: bool,
                 device: torch.device,
                 save_state_path: Optional[Path] = None,
                 load_state_path: Optional[Path] = None,
                 local_save_path: Optional[Path] = None,
                 height: int = 480,
                 width: int = 640,
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
        self.device = device
        self.save_state_path = save_state_path
        self.load_state_path = load_state_path
        self.local_save_path = local_save_path
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
            self.image_buffer = torch.zeros((buffer_size, 3, 3, height, width), dtype=torch.float32, device=self.device)
            self.feature_vector_buffer = torch.zeros((buffer_size, num_features), dtype=torch.float32, device=self.device)
            # Create distance buffer tensor with shape (buffer_size, 3)
            self.distance_buffer = torch.zeros((buffer_size, 3), dtype=torch.float32, device=self.device)

        else: 
            # Read from disk
            raise NotImplementedError

        self.buffer =  {"image_buffer": self.image_buffer,
                        "distance_buffer": self.distance_buffer}

    def add(self, data: Dict[Any, Tensor]):
        """
        This method adds a new image to the replay buffer.
        """

        # Format the data
        data = ContinualSLAMLearner._input_formatter(data)
        if self.count < self.buffer_size:
            if self.save_memory:
                for i in range(self.frame_ids):
                    self.image_buffer[self.count, i] = data[(self.frame_ids[i], 'image')]
                    self.distance_buffer[self.count, i] = data[(self.frame_ids[i], 'distance')]

                    # Get the feature vector of the new image (central image from the image triplet)
                    new_image_feature = self.encoder(data[(0, 'image')])
                    self.feature_vector_buffer[self.count] = new_image_feature
            else:
                # Read from disk
                raise NotImplementedError
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
                    for i in range(self.frame_ids):
                        self.image_buffer[random_index, i] = data[(self.frame_ids[i], 'image')]
                        self.distance_buffer[random_index, i] = data[(self.frame_ids[i], 'distance')]
                        self.feature_vector_buffer[random_index] = new_image_feature
                else:
                    # Read from disk
                    raise NotImplementedError

    def get(self) -> Dict[str, Any]:
        """
        This method returns a random sample of images from the replay buffer.
        """
        if self.count < self.sample_size:
            raise ValueError("The replay buffer does not have enough images to sample from.")
        else:
            if self.save_memory:
                # Sample random indices from the buffer
                random_indices = random.sample(range(self.count), self.sample_size)
                # Get the images and distances from the buffer
                image_sample = self.image_buffer[random_indices]
                distance_sample = self.distance_buffer[random_indices]
            else:
                # Read from disk
                raise NotImplementedError
        
        return {"image_sample": image_sample, "distance_sample": distance_sample}

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
        return self.get()

    def __len__(self):
        return 1

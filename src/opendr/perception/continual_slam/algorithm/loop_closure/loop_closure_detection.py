from pathlib import Path
from typing import List, Tuple

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cosine
from torch import Tensor

from opendr.perception.continual_slam.algorithm.loop_closure.config import LoopClosureDetection as Config
from opendr.perception.continual_slam.algorithm.loop_closure.encoder import LCFeatureEncoder as FeatureEncoder


class LoopClosureDetection:
    def __init__(
        self,
        config: Config,
    ):
        # Initialize parameters ===========================
        self.threshold = config.detection_threshold
        self.id_threshold = config.id_threshold
        self.num_matches = config.num_matches

        # Fixed parameters ================================
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # =================================================

        # Construct network ===============================
        self.model = FeatureEncoder(self.device)
        # =================================================

        # Feature cache ===================================
        # Cosine similarity
        self.faiss_index = faiss.index_factory(self.model.num_features, 'Flat',
                                               faiss.METRIC_INNER_PRODUCT)
        self.image_id_to_index = {}
        self.index_to_image_id = {}
        # =================================================

    def add(self, image_id: int, image: Tensor) -> None:
        # Add batch dimension
        if len(image.shape) == 3:
            image = image.unsqueeze(dim=0)
        features = self.model(image).squeeze().cpu().detach().numpy()
        features = np.expand_dims(features, 0)
        faiss.normalize_L2(features)  # Then the inner product becomes cosine similarity
        self.faiss_index.add(features)
        self.image_id_to_index[image_id] = self.faiss_index.ntotal - 1
        self.index_to_image_id[self.faiss_index.ntotal - 1] = image_id
        # print(f'Is Faiss index trained: {self.faiss_index.is_trained}')

    def search(self, image_id: int) -> Tuple[List[int], List[float]]:
        index_id = self.image_id_to_index[image_id]
        features = np.expand_dims(self.faiss_index.reconstruct(index_id), 0)
        distances, indices = self.faiss_index.search(features, 100)
        distances = distances.squeeze()
        indices = indices.squeeze()
        # Remove placeholder entries without a match
        distances = distances[indices != -1]
        indices = indices[indices != -1]
        # Remove self detection
        distances = distances[indices != index_id]
        indices = indices[indices != index_id]
        # Filter by the threshold
        indices = indices[distances > self.threshold]
        distances = distances[distances > self.threshold]
        # Do not return neighbors (trivial matches)
        distances = distances[np.abs(indices - index_id) > self.id_threshold]
        indices = indices[np.abs(indices - index_id) > self.id_threshold]
        if not len(indices) == 0:
            print(distances)
            print(indices)
            print(index_id)
        # Return best N matches
        distances = distances[:self.num_matches]
        indices = indices[:self.num_matches]
        # Convert back to image IDs
        image_ids = sorted([self.index_to_image_id[index_id] for index_id in indices])
        if not len(indices) == 0:
            print(image_ids)
        return image_ids, distances

    def predict(self, image_0: Tensor, image_1: Tensor) -> float:
        features_0 = self.model(image_0)
        features_1 = self.model(image_1)
        cos_sim = 1 - cosine(features_0.squeeze().cpu().detach().numpy(),
                             features_1.squeeze().cpu().detach().numpy())
        return cos_sim

    @staticmethod
    def display_matches(image_0, image_1, image_id_0, image_id_1, transformation,
                        cosine_similarity):
        # Prevent circular import
        from opendr.perception.continual_slam.algorithm.loop_closure.transform import \
            string_tmat  # pylint: disable=import-outside-toplevel
        if isinstance(image_0, Tensor):
            image_0 = image_0.squeeze().cpu().detach().permute(1, 2, 0)
        if isinstance(image_1, Tensor):
            image_1 = image_1.squeeze().cpu().detach().permute(1, 2, 0)

        filename = Path(f'./figures/sequence_00/matches/{image_id_0:04}_{image_id_1:04}.png')
        filename.parent.mkdir(parents=True, exist_ok=True)

        fig = plt.figure()
        plt.subplot(211)
        plt.imshow(image_0)
        plt.axis('off')
        plt.title(image_id_0)
        plt.subplot(212)
        plt.imshow(image_1)
        plt.axis('off')
        plt.title(image_id_1)
        plt.suptitle(f'cos_sim = {cosine_similarity:.4f} \n {string_tmat(transformation)}')
        plt.savefig(filename)
        plt.close(fig)
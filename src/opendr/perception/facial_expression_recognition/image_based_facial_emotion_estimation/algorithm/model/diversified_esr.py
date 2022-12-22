"""
Implementation of Diversified ESR (Heidari, et al., 2022) trained on AffectNet (Mollahosseini et al., 2017) for facial
expresison recognition.

Code is adapted based on:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks

"""

# Standard libraries
import torch.nn.functional as F
import torch.nn as nn
import torch
import copy
from .cbam import CBAM


class Base(nn.Module):
    """
        The base of the network (Ensembles with Shared Representations, ESRs) is responsible for learning low- and
        mid-level representations from the input data that are shared with an ensemble of convolutional branches
        on top of the architecture.
    """

    def __init__(self):
        super(Base, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)

        # Attention layers
        self.cbam1 = CBAM(gate_channels=64, reduction_ratio=16, pool_types=['avg', 'max'])
        self.cbam2 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'])
        self.cbam3 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'])
        self.cbam4 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'])

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Convolutional, batch-normalization and pooling layers for representation learning
        x_shared_representations = F.relu(self.bn1(self.conv1(x)))
        x_shared_representations, _, _ = self.cbam1(x_shared_representations)

        x_shared_representations = self.pool(F.relu(self.bn2(self.conv2(x_shared_representations))))
        x_shared_representations, _, _ = self.cbam2(x_shared_representations)

        x_shared_representations = F.relu(self.bn3(self.conv3(x_shared_representations)))
        x_shared_representations, _, _ = self.cbam3(x_shared_representations)

        x_shared_representations = self.pool(F.relu(self.bn4(self.conv4(x_shared_representations))))
        x_shared_representations, _, _ = self.cbam4(x_shared_representations)

        return x_shared_representations


class ConvolutionalBranch(nn.Module):
    """
        Convolutional branches that compose the ensemble in ESRs. Each branch was trained on a sub-training
        set from the AffectNet dataset to learn complementary representations from the data (Siqueira et al., 2020).

        Note that, the second last layer provides eight discrete emotion labels whereas the last layer provides
        continuous values of arousal and valence levels.
    """

    def __init__(self):
        super(ConvolutionalBranch, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(128, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 256, 3, 1)
        self.conv3 = nn.Conv2d(256, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)

        # Batch-normalization layers
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.cbam1 = CBAM(gate_channels=128, reduction_ratio=16, pool_types=['avg', 'max'])
        self.cbam2 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'])
        self.cbam3 = CBAM(gate_channels=256, reduction_ratio=16, pool_types=['avg', 'max'])
        self.cbam4 = CBAM(gate_channels=512, reduction_ratio=16, pool_types=['avg', 'max'])

        # Second last, fully-connected layer related to discrete emotion labels
        self.fc = nn.Linear(512, 8)

        # Last, fully-connected layer related to continuous affect levels (arousal and valence)
        self.fc_dimensional = nn.Linear(8, 2)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_shared_representations):
        # Convolutional, batch-normalization and pooling layers
        x_conv_branch = F.relu(self.bn1(self.conv1(x_shared_representations)))
        x_conv_branch, _, _ = self.cbam1(x_conv_branch)

        x_conv_branch = self.pool(F.relu(self.bn2(self.conv2(x_conv_branch))))
        x_conv_branch, _, _ = self.cbam2(x_conv_branch)

        x_conv_branch = F.relu(self.bn3(self.conv3(x_conv_branch)))
        x_conv_branch, _, _ = self.cbam3(x_conv_branch)

        x_conv_branch = F.relu(self.bn4(self.conv4(x_conv_branch)))
        x_conv_branch, attn_ch, attn_sp = self.cbam4(x_conv_branch)  # attn_mat of size 32x1x6x6

        # Prepare features for Classification & Regression
        x_conv_branch = self.global_pool(x_conv_branch)  # N x 512 x 1 x 1
        x_conv_branch = x_conv_branch.view(-1, 512)  # N x 512

        # Fully connected layer for expression recognition
        discrete_emotion = self.fc(x_conv_branch)

        # Fully connected layer for affect perception
        x_conv_branch = F.relu(discrete_emotion)
        continuous_affect = self.fc_dimensional(x_conv_branch)

        return discrete_emotion, continuous_affect, attn_ch, attn_sp


class DiversifiedESR(nn.Module):
    """
    The unified ensemble architecture composed of two building blocks the Base and ConvolutionalBranch
    """

    def __init__(self, device, ensemble_size=9):
        """
        Loads DiversifiedESR.

        :param device: Device to load ESR: GPU or CPU.
        :param ensemble_size: Number of branches

        """

        super(DiversifiedESR, self).__init__()

        # Base of ESR-9 as described in the docstring (see mark 1)
        self.device = device
        self.ensemble_size = ensemble_size

        self.base = Base()
        self.base.to(self.device)

        self.convolutional_branches = []
        for i in range(ensemble_size):
            self.add_branch()

        self.convolutional_branches = nn.Sequential(*self.convolutional_branches)
        self.to(device)

    def get_ensemble_size(self):
        return len(self.convolutional_branches)

    def add_branch(self):
        self.convolutional_branches.append(ConvolutionalBranch())
        self.convolutional_branches[-1].to(self.device)

    def to_state_dict(self):
        state_dicts = [copy.deepcopy(self.base.state_dict())]
        for b in self.convolutional_branches:
            state_dicts.append(copy.deepcopy(b.state_dict()))

        return state_dicts

    def to_device(self, device_to_process="cpu"):
        self.to(device_to_process)
        self.base.to(device_to_process)

        for b_td in self.convolutional_branches:
            b_td.to(device_to_process)

    def reload(self, best_configuration):
        self.base.load_state_dict(best_configuration[0])

        for i in range(self.get_ensemble_size()):
            self.convolutional_branches[i].load_state_dict(best_configuration[i + 1])

    def forward(self, x):
        """
        Forward method of ESR.

        :param x: (ndarray) Input data.
        :return: A list of emotions and affect values from each convolutional branch in the ensemble.
        """

        # List of emotions and affect values from the ensemble
        emotions = []
        affect_values = []
        attn_heads_sp = []
        attn_heads_ch = []

        # Get shared representations
        x_shared_representations = self.base(x)

        # Add to the lists of predictions outputs from each convolutional branch in the ensemble
        for branch in self.convolutional_branches:
            output_emotion, output_affect, attn_ch, attn_sp = branch(x_shared_representations)
            emotions.append(output_emotion)
            affect_values.append(output_affect)
            attn_heads_sp.append(attn_sp[:, 0, :, :])
            attn_heads_ch.append(attn_ch)
        attn_heads_sp = torch.stack(attn_heads_sp)
        attn_heads_ch = torch.stack(attn_heads_ch)
        attn_heads = [attn_heads_sp, attn_heads_ch]

        return emotions, affect_values, attn_heads

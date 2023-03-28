'# Continual Learning 

Continual Learning is a machine learning paradigm enabling a model to learn tasks consecutively without forgetting previously learned knowledge or skills.
In particular, online continual learning operates on a continuous stream of data, i.e., without requiring access to all the data at once.
The goal is to maintain and improve the model's performance over time as it encounters new tasks or domains. 

## Modules

### Continual SLAM: Beyond Lifelong Simultaneous Localization and Mapping Through Continual Learning

For the task of simultaneous localization and mapping (SLAM), Continual SLAM has been included in the OpenDR toolkit.
CL-SLAM leveraging a dual-network architecture to both adapt to new environments and retain knowledge with respect to previously visited environments.
Two separate network heads create predictions for depth estimation and odometry, respectively.
The final combination model creates a 3D pointcloud mapping of the environment based on visual odometry input solely.

Website: http://continual-slam.cs.uni-freiburg.de <br>
Arxiv: https://arxiv.org/abs/2203.01578 <br>
GitHub repository: https://github.com/robot-learning-freiburg/CL-SLAM

**BibTeX**:
```bibtex
@InProceedings{voedisch2023clslam,
  author="V{\"o}disch, Niclas and Cattaneo, Daniele and Burgard, Wolfram and Valada, Abhinav",
  editor="Billard, Aude and Asfour, Tamim and Khatib, Oussama",
  title="Continual SLAM: Beyond Lifelong Simultaneous Localization and Mapping Through Continual Learning",
  booktitle="Robotics Research",
  year="2023",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="19--35",
}
```

**Base repositories**

The OpenDR implementation extends the [Continual SLAM repository](https://github.com/robot-learning-freiburg/CL-SLAM), from [Niclas Vödisch](http://www.informatik.uni-freiburg.de/~voedisch), with the OpenDR interface.

Please note that the original repository is heavily based on
- [monodepthv2](https://github.com/nianticlabs/monodepth2) by the [Niantic Labs](https://www.nianticlabs.com/) authored by [Clément Godard](http://www0.cs.ucl.ac.uk/staff/C.Godard/)

#### Example Usage

More code snippets can be found in [example_usage.py](../../../../projects/python/perception/continual_slam/example_usage.py) with the corresponding [readme](../../../../projects/python/perception/continual_slam/README.md).

**Prepare the downloaded SemanticKITTI dataset** (see the [datasets' readme](./datasets/README.md) as well)
```python
from opendr.perception.continual_slam.datasets import KittiDataset
DOWNLOAD_PATH = '~/data/kitti'
DATA_ROOT = '~/data/kitti'
KittiDataset.prepare_data(DOWNLOAD_PATH, DATA_ROOT)
```

**Run inference**
```python
from opendr.perception.continual_slam.datasets import KittiDataset
from opendr.perception.continual_slam import ContinualSLAMLearner
DATA_ROOT = '~/data/kitti'
config_file = 'configs/singlegpu_kitti.yaml' # stored in continual_slam/configs
predictor = ContinualSLAMLearner(config_file, mode="predictor") # Adapt the path in config folder to load the pretrained model
dataset = KittiDataset(DATA_ROOT, config_file)
for batch in dataset:
    predictions = predictor.infer(batch)
``` 

**Run training**
```python
from opendr.perception.continual_slam.datasets import KittiDataset
from opendr.perception.continual_slam import ContinualSLAMLearner
from opendr.perception.continual_slam.algorithm.depth_pose_module.replay_buffer import ReplayBuffer

DATA_ROOT = '~/data/kitti'
config_file = 'configs/singlegpu_kitti.yaml' # stored in continual_slam/configs
dataset = KittiDataset(DATA_ROOT, config_file)
predictor = ContinualSLAMLearner(config_file, mode="learner") # Adapt the path in config folder to load the pretrained model

replay_buffer = ReplayBuffer(buffer_size=5,
                             save_memory=True,
                             dataset_config_path=config_file,
                             sample_size=3)
for triplet in dataset:
    replay_buffer.add(triplet)
    if replay_buffer.size < 3:
        continue
    triplet = ContinualSLAMLearner._input_formatter(triplet)
    batch = replay_buffer.sample()
    batch.insert(0, triplet)
    learner.fit(batch, learner=True)
```

# Skeleton-based Human Action Recognition
Python implementation of baseline method, ST-GCN [[1]](#1), and the proposed methods 
TA-GCN [[2]](#2), ST-BLN [[3]](#3) and PST-GCN [[4]](#4) for Skeleton-based Human 
Action Recognition. 
The ST-GCN, TA-GCN and ST-BLN methods can be run and evaluated using spatio_temporal_gcn_learner by specifying the model name. 
The PST-GCN method can be run and evaluated using progressive_spatio_temporal_gcn_learner. 

This implementation is adapted from the [OpenMMLAB toolbox](
https://github.com/open-mmlab/mmskeleton/tree/b4c076baa9e02e69b5876c49fa7c509866d902c7).

## Datasets
### NTU-RGB+D-60 
The NTU-RGB+D [[5]](#5) is the largest indoor-captured action recognition dataset which contains different data modalities 
including the $3$D skeletons captured by Kinect-v2 camera. It contains 56,000 action clips from $60$ different action
classes and each action clip is captured by 3 cameras with 3 different views, and provides two different benchmarks,
cross-view (CV) and cross-subject (CS).
In this dataset, the number of joints in each skeleton is 25 and each sample has a sequence of 300 skeletons with 3
different channels each.
### Kinetics-400 
The Kinetics-Skeleton [[6]](#6) dataset is a widely used action recognition dataset which contains the skeleton data of 
300,000 video clips of 400 different actions collected from YouTube. In this dataset each skeleton in a sequence has 18 
joints which are estimated by the OpenPose toolbox [[7]](#7) and each joint is featured by its 2D coordinates and 
confidence score. We used the preprocessed data provided by [[1]](#1) and it can be downloaded from [here](
https://drive.google.com/file/d/103NOL9YYZSW1hLoWmYnv5Fs8mK-Ij7qb/view). 

## References

<a id="1">[1]</a> 
[Yan, S., Xiong, Y., & Lin, D. (2018, April). Spatial temporal graph convolutional networks for skeleton-based action 
recognition. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).](
https://arxiv.org/abs/1609.02907)

<a id="2">[2]</a> 
[Heidari, N., & Iosifidis, A. (2020). Temporal Attention-Augmented Graph Convolutional Network for Efficient Skeleton-
Based Human Action Recognition. arXiv preprint arXiv: 2010.12221.](https://arxiv.org/abs/2010.12221)

<a id="3">[3]</a> 
[Heidari, N., & Iosifidis, A. (2020). On the spatial attention in Spatio-Temporal Graph Convolutional Networks for 
skeleton-based human action recognition. arXiv preprint arXiv: 2011.03833.](https://arxiv.org/abs/2011.03833)

<a id="4">[4]</a> 
[Heidari, N., & Iosifidis, A. (2020). Progressive Spatio-Temporal Graph Convolutional Network for Skeleton-Based Human 
Action Recognition. arXiv preprint arXiv:2011.05668.](https://arxiv.org/pdf/2011.05668.pdf)

<a id="5">[5]</a> 
[Shahroudy, A., Liu, J., Ng, T. T., & Wang, G. (2016). Ntu rgb+ d: A large scale dataset for 3d human activity analysis.
 In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1010-1019).](
 https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html)

<a id="6">[6]</a>
[Kay, W., Carreira, J., Simonyan, K., Zhang, B., Hillier, C., Vijayanarasimhan, S., ... & Zisserman, A. (2017). 
The kinetics human action video dataset. arXiv preprint arXiv:1705.06950.](https://arxiv.org/pdf/1705.06950.pdf) 

<a id="7">[7]</a>
[Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2d pose estimation using part affinity 
fields. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7291-7299).](
https://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html)
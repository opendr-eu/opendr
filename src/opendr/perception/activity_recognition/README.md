# Activity Recognition

This module contains implementations for recent state-of-the-art models for Trimmed Human Activity Recognition on RGB video.


## Models

### X3D

X3D [[ArXiv](https://arxiv.org/abs/2004.04730)][[GitHub](https://github.com/facebookresearch/SlowFast)] is a family of efficient models for video recognition, attaining state-of-the-art performance in offline recognition at multiple accuracy/efficiency trade-offs.

Pretrained X3D models are available [here](https://github.com/facebookresearch/SlowFast/blob/master/MODEL_ZOO.md).

The implementation found in this repository was adapted from the original [source code](https://github.com/facebookresearch/SlowFast).

__Results__:
| Model  |  1-clip       | 10-clip       | 30-clip       |  
| :---:  | :-----------: | :-----------: | :-----------: | 
| X3D-XS | 54.68 (77.52) | 65.59 (86.18) | 65.99 (86.53) |
| X3D-S  | 60.88 (82.52) | 69.97 (89.15) | 70.74 (89.48) |
| X3D-M  | 63.84 (84.27) | 72.51 (90.80) | 73.31 (91.03) |
| X3D-L  | 65.93 (85.60) | 74.37 (91.66) | 74.92 (91.99) |

__BibTeX__:
```bibtex
@article{feichtenhofer2020x3d,
    title={X3D: Expanding Architectures for Efficient Video Recognition},
    author={Christoph Feichtenhofer},
    journal={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020}
}
```



## Datasets

### UCF-101
[UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) is prepared by directly downloading and unpacking [data](http://storage.googleapis.com/thumos14_files/UCF101_videos.zip) and [annotations](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip).

### HMDB-51
[HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) is prepared by directly downloading and unpacking [data](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar) and [annotations](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar).

### Kinetics-400
[Kinetics](https://deepmind.com/research/open-source/kinetics) is a large-scale dataset for Trimmed Human Activity Recognition, consisting of 10 second videos collected from YouTube, ranging over 400 classes.
Due to it's origin, a direct download of the complete dataset is not possible.
Instead, a list of videos and corresponding labels can be downloaded [here](https://storage.googleapis.com/deepmind-media/Datasets/kinetics400.tar.gz), and a [YouTube Crawler](https://github.com/LukasHedegaard/youtube-dataset-downloader) can subsequently be employed to collect the videos one by one. Note: this process may take multiple days.


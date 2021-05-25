# Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images (CVPR 2020)

[Hang Zhou*](https://hangz-nju-cuhk.github.io/), Jihao Liu*, [Ziwei Liu](https://liuziwei7.github.io/), [Yu Liu](http://www.liuyu.us/), and [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/).

[[Paper]](https://arxiv.org/abs/2003.08124)

<img src='./misc/teaser.png' width=880>

We propose a novel un-supervised framework that can synthesize 
photorealistic rotated faces using only single-view image collections 
in the wild. Our key insight is that rotating faces in the 3D space back and forth, 
and re-rendering them to the 2D plane can serve as a strong self-supervision.

  <img src='./misc/pipeline.png' width=800>

## Requirements
* Python 3.6 is used. Basic requirements are listed in the 'requirements.txt'.

```
pip install -r requirements.txt
```

* Install the [Neural_Renderer](https://github.com/daniilidis-group/neural_renderer) following the instructions.


* Download checkpoint and BFM model from [ckpt_and_bfm.zip](https://drive.google.com/file/d/1v31SOrGYueeDi2SxOAUuKWqnglEP0xwA/view?usp=sharing)， put it in ```3ddfa``` and unzip it. Our 3D models are borrowed from [3DDFA](https://github.com/cleardusk/3DDFA). 


## DEMO
### Rotate faces with examples provided
1. Download the [checkpoint](https://drive.google.com/file/d/1Vdlpwghjo4a9rOdn2iJEVlPd0EJegAex/view?usp=sharing)
and put it in ```./checkpoints/rs_model```.

2. Run a simple Rotate-and-Render demo, the inputs are stored at ```3ddfa/example```.

* Modify ```experiments/v100_test.sh```, the ```--poses``` are the desired degrees (range -90 to 90), choose 0 as frontal face.

* Run ```bash experiments/v100_test.sh```, results will be saved at ```./results/```.

## DEVELOP

Prepare your own dataset for testing and training.

### Preprocessing

1. Save the 3D params of human faces to ```3ddfa/results``` by 3ddfa.
```bash
cd 3ddfa
python inference.py --img_list example/file_list.txt --img_prefix example/Images --save_dir results
cd ..
```

### Data Preparation

Modify  ```class dataset_info()``` inside ```data/__ini__.py```, then prepare dataset according to the pattern of the existing example.
You can add the information about a new dataset to each instance of the class.

1. ```prefix``` The absolute path to the dataset.
2. ```file_list``` The list of all images, the absolute path could be incorrect as it is defined in the ```prefix```
3. ```land_mark_list``` The list that stores all landmarks of all images. 
4. ```params_dir``` the path that stores all the 3D params processed before.
5. ```dataset_names``` the dictionary that maps dataset NAMEs to their information. This is used in the parsers as ```--dataset NAME```. 
6. ```folder_level``` the level of folders from the ```prefix``` to images (.jpgs). 
For example the folder_level is 2 if a image is stored as ```prefix/label/image.jpg```.

### Training and Inference
* Modify ```experiments/train.sh``` and use ```bash experiments/train.sh``` for training on new datasets.

* Visualization is supported during training with Tensorboard.

* Please see the DEMO part for inference.

## Details of the Models

We provide two models with trainers in this repo, namely ```rotate``` and ```rotatespade```. The "rotatespade" model is an upgraded one which is different from that described in our paper. A conditional batchnorm module is added according to landmarks predicted from the 3D face. Our checkpoint is trained on this model. We have briefly described this model in our supplementary materials.

## License and Citation
The usage of this software is under [CC-BY-4.0](https://github.com/Hangz-nju-cuhk/Rotate-and-Render/LICENSE).
```
@inproceedings{zhou2020rotate,
  title     = {Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images},
  author    = {Zhou, Hang and Liu, Jihao and Liu, Ziwei and Liu, Yu and Wang, Xiaogang},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020},
}
```

## Acknowledgement
* The structure of this codebase is borrowed from [SPADE](https://github.com/NVlabs/SPADE).
* The [SyncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) module is used in our code.
* We directly borrow the [3DDFA](https://github.com/cleardusk/3DDFA) implementation for 3D reconstruction.

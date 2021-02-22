# Synthentic Multi-view Facial Image Generation based on Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images (CVPR 2020)

[[Paper]](https://arxiv.org/abs/2003.08124)

We cite with small modifications of publicly available code in order to be easily executed, a novel un-supervised framework that can synthesize 
photorealistic rotated faces using only single-view image collections 
in the wild. The key insight is that rotating faces in the 3D space back and forth, 
and re-rendering them to the 2D plane can serve as a strong self-supervision.

#Sources:
* Face Alignment in Full Pose Range: A 3D Total Solution (IEEETPAMI 2017)
* Neural 3D Mesh Renderer (CVPR 2018)
* Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images (CVPR 2020)
## Requirements
* Python 3.6 is used. Basic requirements are listed in the 'requirements.txt'.

```
pip3 install -r requirements.txt
```
* Install the [Neural_Renderer](https://github.com/daniilidis-group/neural_renderer) following the instructions.
```
pip3 install neural_renderer_pytorch
```

* Download checkpoint and BFM model from [ckpt_and_bfm.zip](https://cicloud.csd.auth.gr/owncloud/remote.php/webdav/OpenDR/FTP%20Server%20Material/simulation/ckpt_and_bfm.zip) put it in ```3ddfa``` and unzip it. The 3D models are borrowed from [3DDFA](https://github.com/cleardusk/3DDFA). 


## DEMO

#1. Download the [checkpoint](https://cicloud.csd.auth.gr/owncloud/remote.php/webdav/OpenDR/FTP%20Server%20Material/simulation/latest_net_G.zip)
and put it in ```./checkpoints/rs_model```.

#2.	Clone git the code samples of folder  ```Code ``` in your terminal

## DEVELOP

Prepare the datasets LFW, CelebA, HPID for testing and training. 



### Preprocessing
#3.
a.	Execute at folder  ```3ddfa/Pre-processing``` the following
 ```python3 Do_main_LFW.py ``` which extracts the single view facial texture projected in the 3D head model and extracts the face landmarks features for the candidate faces in of the single view image, for  LFW dataset regulating firstly the respective input/output paths (```rootdir, a.txt, list_lfw_batch.txt```) 

b. Or Execute at folder:  ```3ddfa/Pre-processing ``` the following command  ```python3  Do_main_CelebA.py ``` with the aforementioned function, for CelebA dataset including the respective input/output paths

c. Or Execute at folder:  ```3ddfa/Pre-processing ```the following command  ```python3  Do_main_HPID.py ``` with the aforementioned function, for HPID dataset including the respective input/output paths

#4.	
* Modify ```experiments/v100_test.sh```, the ```--poses``` are the desired degrees (range -90 to 90), choose 0 as frontal face, relative test parameters for generating the facial images in multiple view angles pitch and yaw

* Run ```bash experiments/v100_test.sh```, results will be saved at ```./results/```.



#5.	Results are multi-view facial images for every person identity in a respective folder called  ```results/rs_model/example/```

The respective procedure is followed for your own dataset with minor modifications in ````Pre-processing```` script.


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
* The [SyncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) module is used in the current code.
* We directly borrow the [3DDFA](https://github.com/cleardusk/3DDFA) implementation for 3D reconstruction.

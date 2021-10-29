# Synthentic Multi-view Facial Image Generation based on Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images (CVPR 2020)

Based on: [[Rotate-and-Render: Unsupervised Photorealistic Face Rotation from Single-View Images]](https://arxiv.org/abs/2003.08124)

We utilize with small modifications, in order to be easily executed, publicly available code, namely an un-supervised framework that can synthesize photorealistic rotated facial images using as input  a single facial image, or multiple such images (one per person).
The key insight of the utilized method is that rotating faces in the 3D space back and forth, and re-rendering them to the 2D plane can serve as a strong self-supervision.

# Sources:
* Face Alignment in Full Pose Range: A 3D Total Solution (IEEE TPAMI 2017)
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

* Download checkpoint and BFM model from [ckpt_and_bfm.zip](https://155.207.128.10/owncloud/remote.php/webdav/CIDL/OpenDR_internal/Rotate_and_Render/Code/checkpoints/ckpt_and_bfm.zip) put it in ```3ddfa``` and unzip it. The 3D models are borrowed from [3DDFA](https://github.com/cleardusk/3DDFA). 


## DEMO

1. Download the [checkpoint](https://cicloud.csd.auth.gr/owncloud/remote.php/webdav/OpenDR/FTP%20Server%20Material/simulation/latest_net_G.zip)
and put it in ```./checkpoints/rs_model```.

2.	Execute the one-step OPENDR function 3ddfa/testSyntheticDataGeneration.py specifying the input images folder, the output folder, the desired degrees (range -90 to 90) for generating the facial images in multiple view angles pitch and yaw as indicated in the command line: 
```sh
cd 3ddfa

python3 testSyntheticDataGeneration.py
```
3. Results are multi-view facial images for every person identity in a respective folder called  ```results/rs_model/example/```


## License and Citation
The usage of this software is under [CC-BY-4.0](https://github.com/Hangz-nju-cuhk/Rotate-and-Render/blob/master/LICENSE).


## Acknowledgement
Large parts of the code are taken from: 
* The structure of this codebase is borrowed from [SPADE](https://github.com/NVlabs/SPADE).
* The [SyncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) module is used in the current code.
* The [3DDFA](https://github.com/cleardusk/3DDFA) implementation for 3D reconstruction.
* The code [Rotate-and-Render](https://github.com/Hangz-nju-cuhk/Rotate-and-Render/)  
  
with the following modifications to make them compatible with the OpenDR specifications:
## Minor Modifications
1. All scripts: PEP8 changes
2. ```3ddfa/main.py, 3ddfa/inference.py, test_multipose.py``` Modified to work as a callable functions
3. ```options/base_options.py, options/test_options.py ``` Commented out/change several parameters to easily executed 
4. ```models/networks/render.py``` Minor functional changes
5. The OPENDR created functions are ```3ddfa/SyntheticDataGeneration.py, 3ddfa/testSyntheticDataGeneration.py, 3ddfa/path_helper.py, 3ddfa/path_helper2.py```
6. The rest are taken from the aforementioned repositories

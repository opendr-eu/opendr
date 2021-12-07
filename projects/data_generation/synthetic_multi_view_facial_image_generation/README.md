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

* Download checkpoint and BFM model from [ckpt_and_bfm.zip](ftp://opendrdata.csd.auth.gr/data_generation/synthetic_multi-view-facial-generator/checkpoints.zip) put it in ```3ddfa``` and unzip it. The 3D models are borrowed from [3DDFA](https://github.com/cleardusk/3DDFA). 

## DEMO

1. Download the [checkpoint](ftp://opendrdata.csd.auth.gr/data_generation/synthetic_multi-view-facial-generator/checkpoints.zip)
and put it in ```./checkpoints/rs_model```.

2.	Execute the one-step OPENDR function tool_synthetic_facial_generation.py specifying the input images folder, the output folder, the desired degrees (range -90 to 90) for generating the facial images in multiple view angles pitch and yaw as indicated in the command line: 
```sh
python3 tool_synthetic_facial_generation.py -path_in ./demos/imgs_input/ -path_3ddfa ./algorithm/DDFA/ -save_path ./results -val_yaw 10, 40 -val_pitch 10, 30 -device cuda

```
3. Results are multi-view facial images for every person identity in a respective folder called  ```results/rs_model/example/```


## License 
The usage of this software code Rotate-and-Render is under [CC-BY-4.0](https://github.com/Hangz-nju-cuhk/Rotate-and-Render/blob/master/LICENSE).
The usage of this software code SPADE, SyncBN, 3DDFA are under [MIT License](https://github.com/tasostefas/opendr_internal/blob/synthetic-multi-view-facial-generator/projects/data_generation/synthetic-multi-view-facial-image-generation/3ddfa/LICENSE)

## Acknowledgement
Large parts of the code are taken from: 
* The structure of this codebase is borrowed from [SPADE](https://github.com/NVlabs/SPADE).
* The [SyncBN](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch) module is used in the current code.
* The [3DDFA](https://github.com/cleardusk/3DDFA) implementation for 3D reconstruction.
* The code [Rotate-and-Render](https://github.com/Hangz-nju-cuhk/Rotate-and-Render/)  
  
with the following modifications to make them compatible with the OpenDR specifications:
## Minor Modifications
1. All scripts: PEP8 changes
2. ```3ddfa/preprocessing_1.py, 3ddfa/preprocessing_2.py, test_multipose.py``` Modified to work as a callable functions
3. ```options/base_options.py, options/test_options.py ``` Commented out/change several parameters to be easily executed 
4. ```models/networks/render.py``` Minor functional changes
5. The OPENDR created functions are ```3ddfa/SyntheticDataGeneration.py, 3ddfa/testSyntheticDataGeneration.py```
6. The rest are taken from the aforementioned repositories

# Skeleton-based Human Action Recognition Demo

This folder contains an implemented demo of skeleton_based human action recognition method provided by the [OpenDR toolkit](https://opendr.eu).
This demo employs lightweight OpenPose method provided by OpenDR toolkit to extract the human body skeletons from the videos, and it can be performed by either loading an action video file as input or capturing the video stream from a webcam. 

#### Data preparation  
  Download the NTU-RGB+D video data from [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/), 
  then run the following function to extract the skeleton data using the lightweight OpenPose method: 
  ```python
python3 skeleton_extraction.py --videos_path path_to_dataset --out_folder path_to_save_skeleton_data --device cuda --num_channels 2  
```  
  `--num_channels` specifies the number of dimensions for each body keypoint. The lightweight OpenPose extracts 2 dimensional keypoints, denoting the keypoint coordinates, and it also provides the confidence score for each keypoint which can be used as the third dimension. 
  
After generating the 2d or 3d skeleton data, you need to train a model on this dataset using the implemented learners, `spatio_temporal_gcn_learner` or `progressive_spatio_temporal_gcn_learner` and use the pre-trained models for running the demo. 

#### Running demo
We provided the pre-trained models of ST-GCN and PST-GCN methods and the demo can be run as follows: 

```python
python3 demo.py --device cuda --video 0 --method stgcn --action_checkpoint_name stgcn_ntu_cv_lw_openpose
```  

Please use the `--device cpu` flag if you are running them on a machine without a CUDA-enabled GPU. 

```python
python3 demo.py --device cpu --video 0 --method pstgcn --action_checkpoint_name pstgcn_ntu_cv_lw_openpose
```  

A demo script for the Continual ST-GCN can be found in [costgcn_usage_demo.py](./demos/costgcn_usage_demo.py). To fit, evaluate, and run inference on the model, you may use the following command:
```bash
python3 costgcn_usage_demo.py --fit --eval --infer
```  

## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.

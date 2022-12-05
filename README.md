# Voxel Pseudo Image Tracking(VPIT) + OpenDR Toolkit

This repository contains a Pytorch implementation of [VPIT: Real-time Embedded Single Object 3D Tracking Using Voxel Pseudo Images](https://arxiv.org/abs/2206.02619) paper inside of an OpenDR toolkit.

In this paper, we propose a novel voxel-based 3D single object tracking (3D SOT) method called Voxel Pseudo Image Tracking (VPIT). VPIT is the first method that uses voxel pseudo images for 3D SOT. The input point cloud is structured by pillar-based voxelization, and the resulting pseudo image is used as an input to a 2D-like Siamese SOT method. The pseudo image is created in the Bird's-eye View (BEV) coordinates, and therefore the objects in it have constant size. Thus, only the object rotation can change in the new coordinate system and not the object scale. For this reason, we replace multi-scale search with a multi-rotation search, where differently rotated search regions are compared against a single target representation to predict both position and rotation of the object. Experiments on KITTI Tracking dataset show that VPIT is the fastest 3D SOT method and maintains competitive Success and Precision values. Application of a SOT method in a real-world scenario meets with limitations such as lower computational capabilities of embedded devices and a latency-unforgiving environment, where the method is forced to skip certain data frames if the inference speed is not high enough. We implement a real-time evaluation protocol and show that other methods lose most of their performance on embedded devices, while VPIT maintains its ability to track the object.

If you use this work for your research, you can cite it as:
```
@article{oleksiienko2022vpit,
  author = {Oleksiienko, Illia and Nousi, Paraskevi and Passalis, Nikolaos and Tefas, Anastasios and Iosifidis, Alexandros},
  journal={arxiv:2206.02619}, 
  title={VPIT: Real-time Embedded Single Object 3D Tracking Using Voxel Pseudo Images}, 
  year={2022},
}
```

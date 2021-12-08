**Todo:** Update the neck-removing processing pipeline from original BFM model.

The original version with neck:
<p align="center">
  <img src="imgs/bfm_noneck.jpg" alt="neck" width="400px">
</p>

[bfm.ply](https://github.com/Hangz-nju-cuhk/Rotate-and-Render/blob/master/3ddfa/BFM_Remove_Neck/bfm.ply)

The refined version without neck:
<p align="center">
  <img src="imgs/bfm_refine.jpg" alt="no neck" width="400px">
</p>

[bfm_refine.ply](https://github.com/Hangz-nju-cuhk/Rotate-and-Render/blob/master/3ddfa/BFM_Remove_Neck/bfm_refine.ply)
These two images are rendered by MeshLab.

`bfm_show.m` shows how to render it with 68 keypoints in Matlab.

<p align="center">
  <img src="imgs/bfm_refine.jpg" alt="no neck">
</p>

Attention: the z-axis value of `bfm.ply` and `bfm_refine.ply` file are opposed in `model_refine.mat`, do not use these two `ply` file in training.

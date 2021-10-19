**Todo:** Update the neck-removing processing pipeline from original BFM model.

The original version with neck:
[bfm.ply](https://155.207.128.10/owncloud/remote.php/webdav/SFTP/CIDL/OpenDR_internal/Rotate_and_Render/Code/3ddfa/BFM_Remove_Neck/bfm.ply)

The refined version without neck:
[bfm_refine.ply](https://155.207.128.10/owncloud/remote.php/webdav/SFTP/CIDL/OpenDR_internal/Rotate_and_Render/Code/3ddfa/BFM_Remove_Neck/bfm_refine.ply)
These two images are rendered by MeshLab.

`bfm_show.m` shows how to render it with 68 keypoints in Matlab.

<p align="center">
  <img src="imgs/bfm_refine.jpg" alt="no neck">
</p>

Attention: the z-axis value of `bfm.ply` and `bfm_refine.ply` file are opposed in `model_refine.mat`, do not use these two `ply` file in training.

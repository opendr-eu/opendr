# SMPL+D body models

This folder contains code for downloading a large number of human models in SMPL+D body model format, as well as code for making them able to be transferred to Webots for simulations.

## Download the raw SMPL+D models.
```
python download_smpld_raw.py
```

## Download the SMPL+D models as FBX (Filmbox) files. <br/>In this case the pose blend shapes are not applied.
** Download SMPL for MAYA from the official website [here](https://smpl.is.tue.mpg.de/).<br/>
Once you agree on SMPL license terms and have access to downloads, you will have the following two files:
```
basicModel_f_lbs_10_207_0_v1.0.2.fbx
basicModel_m_lbs_10_207_0_v1.0.2.fbx
```
Place these two files in the model directory.

```
python download_smpld_fbx.py
```

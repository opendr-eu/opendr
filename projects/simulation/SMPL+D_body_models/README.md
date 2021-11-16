# SMPL+D body models

This folder contains code for downloading a large number of human models in SMPL+D body model format, as well as code for making them able to be transferred to Webots for simulations.

## Download the raw SMPL+D models.
```
python download_data.py raw
```

## Download the SMPL+D models as FBX (Filmbox) files. <br/>In this case the pose blend shapes are not applied.

- Download the data by running:
```
python src/download_data.py
```

- Download SMPL for Unity from the official website [here](https://smpl.is.tue.mpg.de/). Once you agree on SMPL license terms and have access to downloads, you will have the following two files:
```
SMPL_f_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx
SMPL_m_unityDoubleBlends_lbs_10_scale5_207_v1.0.0.fbx
```
- Place these two files in the ```model``` directory.

-Generate the human models as FBX files. Export the path to Blender before generating the models.
```
export BLENDER_PATH=path_to_Blender
$BLENDER_PATH/blender -P make_fbx.py
```

# SMPL+D body models

This folder contains code for downloading a large number of human models in SMPL+D body model format, as well as code for making them able to be transferred to Webots for simulations.

<p float="left">
  <img src="./examples/model_1.png" width=150 />
  <img src="./examples/model_4.png" width=150 />
  <img src="./examples/model_3.png" width=150 />
  <img src="./examples/model_2.png" width=150 />
</p>

### Download the raw SMPL+D models only.
```
python download_data.py raw
```

### Download the SMPL+D models as FBX (Filmbox) files. <br/>In this case the pose blend shapes are not applied.

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

- Download and install Blender (tested on 2.93.4 version)

- Set the path to Blender.
```
export BLENDER_PATH=path_to_Blender
```
- Install the necessary packages to Blender's Python using pip (example using Blender 2.93 and Python 3.9):
```
$BLENDER_PATH/2.93/python/bin/python3.9 -m ensurepip
$BLENDER_PATH/2.93/python/bin/python3.9 -m pip install numpy opencv-python opencv-contrib-python scipy
```
- Generate the human models as FBX files. 
```
$BLENDER_PATH/blender -P src/generate_models.py
```

### Install a demo Webots project for animating SMPL+D models.

-  Download a database from AMASS (https://amass.is.tue.mpg.de/download.php)
-  Extract the database (e.g., tar -xf ACCAD) 
-  Run:
```
python webots/extract_anims.py _ path_to_database _
```
-  Assign the directory of the selected animation in the controllerArgs to webots/smpl_webots/worlds/demo_world.wbt
-  Install the demo project in Webots
```
chmod +x install_project.sh
bash install_project.sh
```

# Human Model Generation Demos

This folder contains a jupyter notebook demo that demonstrates the various functionalities of the Human Model Generation module provided by OpenDR.

More specifically, the ```demos/inference_tutorial.ipynb```  is a step-by-step tutorial on how to:

1. Generate a 3D human model from a single image using the```PIFuGeneratorLearner```.
2. Extract the 3D model's 3D pose.
3. Get renderings of the 3D model from various views.
4. Visualize the renderings.

As input, an image from the [Clothing Co-Parsing (CCP)](https://github.com/bearpaw/clothing-co-parsing) dataset is provided in the ```demos/imgs_input/rgb``` folder, as well as an image of the silhouette of the depicted human (mask) in the ```demos/imgs_input/msk folder```.
Please use the `--device cpu` flag for the demos if you are running them on a machine without a CUDA-enabled GPU. 

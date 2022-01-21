## semantic_segmentation module

The *semantic segmentation* module contains the *BisenetLearner* class, which inherit from the abstract class *Learner*.


### Class BisenetLearner
Bases: `engine.learners.Learner`

The *BisenetLearner* class is a wrapper of the BiseNet model [[1]](#bisenetp) found on [BiseNet] (https://github.com/ooooverflow/BiSeNet).
It is used to train Semantic Segmentation models on RGB images and run inference.
The [BisenetLearner](/src/opendr/perception/semantic_segmentation/bisenet/bisenet_learner.py) class has the following public methods:

#### `BisenetLearner` constructor
```python
BisenetLearner(self, lr, iters, batch_size, optimizer, temp_path, checkpoint_after_iter, checkpoint_load_iter, device, val_after, weight_decay, momentum, drop_last, pin_memory, num_workers, num_classes, crop_height, crop_width, context_path)
```

Constructor parameters:

  - **lr**: *float, default=0.01*\
    Learning rate during optimization.
  - **iters**: *int, default=1*\
    Number of epochs to train for.
  - **batch_size**: *int, default=1*\
    Dataloader batch size. Defaults to 1.
  - **optimizer**: *str, default="sgd"*\
    Name of optimizer to use ("sgd" ,"rmsprop", or "adam").
  - **temp_path**: *str, default=''*\
    Path in which to store temporary files.
  - **checkpoint_after_iter**: *int, default=0*\
    Save chackpoint after specific epochs.
  - **checkpoint_load_iter**: *int, default=0*\
    Unused parameter.
  - **device**: *str, default="cpu"*\
    Name of computational device ("cpu" or "cuda").
  - **val_after**: *int, default=1*\
    Perform validation after specific epochs.
  - **weight_decay**: *[type], default=5e-4*\
    Weight decay used for optimization.
  - **momentum**: *float, default=0.9*\
    Momentum used for optimization.
  - **drop_last**: *bool, default=True*\
    Drop last data point if a batch cannot be filled.
  - **pin_memory**: *bool, default=False*\
    Pin memory in dataloader.
  - **num_workers**: *int, default=4*\
    Number of workers in dataloader.
  - **num_classes**: *int, default=12*\
    Number of classes to predict among.
  - **crop_height**: *int, default=720*\
    Input image height.
  - **crop_width**: *int, default=960*\
    Input image width.
  - **context_path**: *str, default='resnet18'*\
    Context path for the bisenet model.


#### `BisenetLearner.fit`
```python
BisenetLearner.fit(self, dataset, val_dataset, silent, verbose)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.

Parameters:
  - **dataset**: *Dataset*\
    Training dataset.
  - **val_dataset**: *Dataset, default=None*\
    Validation dataset. If none is given, validation steps are skipped.
  - **silent**: *bool, default=False*\
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=True*\
    If set to True, enables the maximum logging verbosity.


#### `BisenetLearner.eval`
```python
BisenetLearner.eval(self, dataset, silent, verbose)
```
This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.

Parameters:
  - **dataset**: *Dataset*\
    Dataset on which to evaluate model.
  - **silent**: *bool, default=False*\
    If set to True, disables all printing of training progress reports and other information to STDOUT.
  - **verbose**: *bool, default=True*\
    If set to True, enables the maximum logging verbosity.


#### `BisenetLearner.infer`
```python
BisenetLearner.infer(self, img)
```

This method is used to perform segmentation on an image.
Returns a `engine.target.Heatmap` object.

Parameters:
  - **img**: *Image*\
    Image to predict a heatmap.


#### `BisenetLearner.download`
```python
BisenetLearner.download(self, path, mode, verbose, url)
```

Download pretrained models and testing images to path.

Parameters:
- **path**: *str, default=None*\
  Path to metadata file in json format or to weights path.
- **mode**: *{'pretrained', 'testingImage'}, default='pretrained'*\
  If *'pretrained'*, downloads a pretrained segmentation model. If *'testingImage'*, downloads an image to perform inference on.
- **verbose**: *bool, default=True*\
  If True, enables maximum verbosity.
- **url**: *str, default=OpenDR FTP URL*\
  URL of the FTP server.


#### `BisenetLearner.save`
```python
BisenetLearner.save(self, path, verbose)
```

Save model weights and metadata to path.

Parameters:
- **path**: *str*\
  Directory in which to save model weights and metadata.
- **verbose**: *bool, default=True*\
  If set to True, enables the maximum logging verbosity.


#### `BisenetLearner.load`
```python
BisenetLearner.load(self, path)
```

This method is used to load a previously saved model from its saved folder.

Parameters:
- **path**: *str*\
  Local path to save the files.


#### Examples

* **Training example on CamVid train set.**
  ```python
  import os
  from opendr.perception.semantic_segmentation import BisenetLearner
  from opendr.perception.semantic_segmentation import CamVidDataset


  if __name__ == '__main__':
      learner = BisenetLearner()
      # Download CamVid dataset
      if not os.path.exists('./datasets/'):
          CamVidDataset.download_data('./datasets/')
      datatrain = CamVidDataset('./datasets/CamVid/', mode='train')
      learner.fit(dataset=datatrain)
      learner.save("./bisenet_saved_model")
  ```

* **Evaluation example on CamVid test set.**
  ```python
  import os
  from opendr.perception.semantic_segmentation import BisenetLearner
  from opendr.perception.semantic_segmentation import CamVidDataset


  if __name__ == '__main__':
      learner = BisenetLearner()

      # Download CamVid dataset
      if not os.path.exists('./datasets/'):
          CamVidDataset.download_data('./datasets/')
      datatest = CamVidDataset('./datasets/CamVid/', mode='test')

      # Download the pretrained model
      learner.download('./bisenet_camvid', mode='pretrained')
      learner.load('./bisenet_camvid')
      results = learner.eval(dataset=datatest)

      print("Evaluation results = ", results)
  ```

* **Inference example on a single test image using a pretrained model.**
  ```python
  import cv2
  from opendr.perception.semantic_segmentation import BisenetLearner
  from opendr.engine.data import Image
  from matplotlib import cm
  import numpy as np

  if __name__ == '__main__':
      learner = BisenetLearner()

      # Dowload the pretrained model
      learner.download('./bisenet_camvid', mode='pretrained')
      learner.load('./bisenet_camvid')

      # Download testing image
      learner.download('./', mode='testingImage')
      img = Image.open("./test1.png")

      # Perform inference
      heatmap = learner.infer(img)

      # Create a color map and translate colors
      segmentation_mask = heatmap.data

      colormap = cm.get_cmap('viridis', 12).colors
      segmentation_img = np.uint8(255*colormap[segmentation_mask][:, :, :3])

      # Blend original image and the segmentation mask
      blended_img = np.uint8(0.4*img.opencv() + 0.6*segmentation_img)

      cv2.imshow('Heatmap', blended_img)
      cv2.waitKey(-1)
  ```
  
  
#### References
<a name="bisenetp" href="https://arxiv.org/abs/1808.00897">[1]</a> BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation,
[arXiv](https://arxiv.org/abs/1808.00897).

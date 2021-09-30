OpenDR 2D Face Detection - RetinaFace
======

This folder contains the OpenDR Learner class for RetinaFace for 2D face detection.

Sources
------
Large parts of the Learner code are taken from the official [deepinsight implementation](https://www.github.com/deepinsight/insightface) with modifications to make it compatible with OpenDR specifications. The original code is licensed under the MIT license:
```
MIT License

Copyright (c) 2018 Jiankang Deng and Jia Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Usage
------
- ```make``` should be run from the parent directory to build some sources.
- Only ```resnet``` network is supported for training, and only the WIDER Face dataset can be used for training.
- The ```temp_path``` folder is used to save checkpoints during training, as well as a ```.pkl``` file which is generated during training and contains the roidb.
- To train the detector, some extra data must be downloaded and placed in the directory of the WIDER Face dataset. The directory structure of the dataset should look like:
  
  ``` 
  root
  └─── WIDER_train
  |    |   label.txt
  |    └───images
  └─── WIDER_val
  |    |   label.txt
  |    └───images
  └─── WIDER_test
  |    |   label.txt
  |    └───images
  └─── wider_face_split
          wider_face_train_bbx_gt.txt
          wider_face_val_bbx_gt.txt
  ```
  The data is downloaded automatically during the training process. See ```train_demo.py``` for details.
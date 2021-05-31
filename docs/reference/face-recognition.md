## face_recognition_learner Module

The *face_recognition_learner* module contains the FaceRecognitionLearner class, which inherits from the abstract method *Learner*.

### Class FaceRecognitionLearner
Bases: `engine.learners.Learner`

FaceRecognition class is OpenDR's implementation for training and using a model on the face recognition task.

The [FaceRecognition](#src.opendr.perception.face_recognition.face_recognition_learner.py) class has the
following public methods:

#### `FaceRecognitionLearner` constructor
```python
FaceRecognitionLearner(self, lr, iters, batch_size, optimizer, device, threshold, backbone, network_head, loss, temp_path, mode, checkpoint_after_iter, checkpoint_load_iter, val_after, input_size, rgb_mean, rgb_std, embedding_size, weight_decay, momentum, drop_last, stages, pin_memory, num_workers, seed)
```

Constructor parameters:
- **lr**: *float, default=0.1*  
  Specifies the initial learning rate to be used during training.
- **iters**: *int, default=120*  
  Specifies the number of iterations the training should run for.
- **batch_size**: *int, default=128*  
  Specifies number of images to be bundled up in a batch during training. This heavily affects memory usage, adjust according to your system.
- **optimizer**: *{'sgd'}, default='sgd'*  
  Specifies the optimizer to be used during training. Currently supports 'sgd' (stochastic gradient decent).
- **device**: *{'cpu', 'cuda'}, default='cuda'*  
  Specifies the device to be used.
- **threshold**: *float, default=0.0*  
  The backbone's threshold for accepting a positive match during inference. This is set when a pretrained is loaded, but the user can specify a different threshold.
- **backbone**: *{'resnet_50, 'resnet_101', 'resnet_152', 'ir_50', 'ir_101', 'ir_152', 'ir_se_50', 'ir_se_101', 'ir_se_152', 'mobilefacenet'}, default='ir_50'*  
  Specifies the backbone architecture.
- **network_head**: *{'arcface, 'cosface', 'sphereface', 'am_softmax', 'classifier'}, default='arcface'*  
  Specifies the head architecture.
- **loss**: *{focal}, default='focal'*  
  Specifies the loss to be used during training.
- **temp_path**: *str, default='./temp'*  
  Specifies a path where the algorithm looks for pretrained backbone weights, the checkpoints are saved along with the logging files.
- **mode**: *{'backbone_only, 'head_only', 'full'}, default='backbone_only'*  
  The module supports four modes:
  - 'backbone_only':  
    Used for inference. Only the backbone architecture is used, and inference is done through retrieval.  
  - 'head_only':  
    Used for training a classifier head. Only the head architecture will be trained, using a pretrained backbone model.
  - 'full':  
    Used to train a combined model (backbone + head) or to inference with a classifier head.
  - 'finetune':  
    Used to finetune a previously trained backbone. It loads a saved backbone model and creates a new head to finetune the backbone model.
- **checkpoint_after_iter**: *int, default=0*  
  Specifies after how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved. 
- **checkpoint_load_iter**: *int, default=0*  
  Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.
- **val_after**: *int, default=0*  
  Specifies after how many training iterations a validation should be run.          
- **input_size**: *list, default=[112, 112]*  
  Specifies the input size of the images.         
- **rgb_mean**: *list, default=[0.5, 0.5, 0.5]*  
  Specifies the mean values with which the input should be normalized.  
- **rgb_std**: *list, default=[0.5, 0.5, 0.5]*  
  Specifies the standard deviation values with which the input should be normalized.  
- **embedding_size**: *int, default=512*  
  Specifies the size of the backbone's output.
- **weight_decay**: *float, default=5e-4*  
  Specifies the rate the weights should be decayed by the optimizer during training. 
- **momentum**: *float, default=0.9*  
  Specifies the momentum used by the optimizer during training.
- **drop_last**: *bool, default=True*  
  Specifies whether the last batch should be dropped or not if it is not complete.
- **stages**: *list, default=[35, 65, 95]*  
  Specifies on which training iterations the learning rate should be adjusted. The learning rate will get divided by 10 on each of those stages.
- **pin_memory**: *bool, default=True*  
  Specifies if pinned memory should be used by the Dataloader.
- **num_workers**: *int, default=4*  
  Specifies the number of workers to be used by the Dataloader.


#### `FaceRecognitionLearner.fit`
```python
FaceRecognitionLearner.fit(self, dataset, val_dataset, logging_path, silent, verbose)
```

This method is used for training the algorithm on a train dataset and validating on a val dataset.
Returns a dictionary containing stats regarding the last evaluation ran.  

Parameters: 
  
- **dataset**: *object*  
  Object that holds the training dataset.  
- **val_dataset**: *object, default=None*  
  Object that holds the validation dataset.  
- **logging_path**: *str, default=''*  
  Path to save tensorboard log files.  
  If set to None or ‘’, tensorboard logging is disabled
- **silent**: *bool, default=False*  
  If set to True, disables all printing of training progress reports and other information to STDOUT.  
- **verbose**: *bool, default=True*  
  If set to True, enables the maximum logging verbosity.  


**Notes**

Train dataset should be of type 'ImageFolder'. Images should be placed in a defined structure like:
- imgs
    - ID1
      - image1
      - image2
    - ID2
    - ID3
    - ...

When training a classifier head, the 'val_dataset' should also be of type 'Imagefolder' of the same classes as the training dataset since the model is trained to predict specific classes.

#### `FaceRecognitionLearner.eval`
```python
FaceRecognitionLearner.eval(self, dataset, num_pairs, silent, verbose)
```

This method is used to evaluate a trained model on an evaluation dataset.
Returns a dictionary containing stats regarding evaluation.  
Parameters:

- **dataset**: *object, default=None*  
  Object that holds the evaluation dataset.
  Can be of type `ExternalDataset` or a custom dataset inheriting from `DatasetIterator`.  
  Supports the following datasets:  
  - 'lfw' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch))
  - 'cfp_ff' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch))
  - 'cfp_fp' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch))
  - 'agedb_30' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch))
  - 'vgg2_fp' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch))
  - 'imagefolder' (random positive and negative pairs will be created to perform the evaluation)
- **num_pairs**: *int, defailt=1000*  
  When an 'ImageFolder' dataset is used for evaluation, sets the number of random pairs, positive and negative, that will be created.
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.
- **verbose**: *bool, default=True*  
  If set to True, enables the maximum verbosity.

**Notes**

When 'ExternalDataset' is of type 'ImageFolder', images should be placed in a defined structure like:
- imgs
    - ID1
      - image1
      - image2
    - ID2
    - ID3
    - ...


#### `FaceRecognitionLearner.fit_reference`
```python
FaceRecognitionLearner.fit_reference(self, path, save_path)
```

This method is used to create a reference database to be used in inference when mode='backbone_only'.
It creates a pickle file containing a dictionary of ID - embedding. If more than one image is used for each ID, the average embedding is kept.

Parameters:  
- **path**: *str, default=None*  
  Path containing the reference images. If a reference database was already created can be left blank.  
- **save_path**: *str, default=None*  
  Path to save (load if already created) the .pkl reference file.

**Notes**

Reference images should be placed in a defined structure like:
- imgs
    - ID1
      - image1
      - image2
    - ID2
    - ID3
    - ...
    
When calling infer the method returns the name of the sub-folder, e.g. ID1.


#### `FaceRecognitionLearner.infer`
```python
FaceRecognitionLearner.infer(self, img)
```

This method is used to perform face recognition on an image.
Returns a `engine.target.Category` object, or returns None if no entry in the reference database was closer than the set threshold (when in backbone_only mode). 

Parameters:
- **img**: *object*  
  Object of type 'engine.data.Image'.
  


#### `FaceRecognitionLearner.align`
```python
FaceRecognitionLearner.align(self, data, dest, crop_size, silent)
```

This method is used for aligning the faces in an 'ImageFolder' dataset.
Face recognition algorithms return better results when the images fed for inference contain only a face, centered and aligned.

Parameters:

- **data**: *str, default=''*  
  The folder containing the images to be aligned.  
- **dest**: *str, default='/aligned' 
  The destination folder to save the aligned images.
- **crop_size**: *int, default=112*  
  The size of the produced images (crop_size x crop_size).
- **silent**: *bool, default=False*  
  If set to True, disables all printing of evaluation progress reports and other information to STDOUT.


#### `FaceRecognitionLearner.save`
```python
FaceRecognitionLearner.save(self, path)
```

This method is used to save a trained model.
Provided with the path '/my/path/' (absolute or relative), it creates the directory, if it does not already 
exist.  
Inside this folder, the model is saved as 'backbone.pth' and the metadata file as "backbone.json".  
If the directory already exists, the existing .pth and .json files are overwritten.

If [`self.optimize`](#FaceRecognitionLearner.optimize) was run previously, it saves the optimized ONNX model in a similar fashion with an ".onnx" extension, by copying it from the self.temp_path it was saved previously during conversion.

Parameters:
- **path**: *str, default=None*  
  Path to save the model.


#### `FaceRecognitionLearner.load`
```python
FaceRecognitionLearner.load(self, path)
```

This method is used to load a previously saved model from its saved folder.
Loads the model from inside the directory of the path provided, using the metadata .json file included.

Parameters:
- **path**: *str, default=None*  
  Path of the model to be loaded.


#### `FaceRecognitionLearner.optimize`
```python
FaceRecognitionLearner.optimize(self, do_constant_folding)
```

This method is used to optimize a trained model to ONNX format which can be then used for inference.

Parameters:
- **do_constant_folding**: *bool, default=False*  
  ONNX format optimization.
  If True, the constant-folding optimization is applied to the model during export.  
  Constant-folding optimization will replace some of the ops that have all constant inputs, with pre-computed constant nodes.
  


#### `FaceRecognitionLearner.download`
```python
FaceRecognitionLearner.download(self, path, mode)
```

Download utility for various Face Recognition components.  
Downloads files depending on mode and saves them in the path provided.  
It supports downloading:  
1. pretrained models (currently supporting mobilefacenet and ir_50)
2. images to run tests on.

Parameters:
- **path**: *str, default=None*  
  Local path to save the files, defaults to self.temp_path if None.
- **mode**: *str, default="pretrained"*  
  Which files to download, can be one of "pretrained", "test_data".
  

#### Examples

* **Training example using an 'ExternalDataset'**.

```python
from opendr.perception.face_recognition.face_recognition_learner import FaceRecognitionLearner
from opendr.engine.datasets import ExternalDataset
recognizer = FaceRecognitionLearner(backbone='ir_50', mode='full', network_head='arcface',
                                    epochs=120, lr=0.1, checkpoint_after_iter=10, checkpoint_load_iter=0,
                                    device='cuda', val_after=40)
train = ExternalDataset(path='./data', dataset_type='imagefolder')
evaluation = ExternalDataset(path='./data', dataset_type='lfw')
recognizer.fit(dataset=train, val_dataset=evaluation)
recognizer.save('./temp/saved_models')
```


* **Inference example - backbone_only mode**
```python
import cv2
from opendr.perception.face_recognition.face_recognition_learner import FaceRecognitionLearner
recognizer = FaceRecognitionLearner(backbone='ir_50', mode='backbone_only', device='cuda')
recognizer.load('./temp/saved_models')
recognizer.fit_reference(path='./data/imgs', save_path='./temp/demo')
img = cv2.imread('test.jpg')
result = recognizer.infer(img)
print(result)
```



* **Inference example - camera feed**
```python
import cv2
import os
import numpy as np
import time
from opendr.perception.face_recognition.face_recognition_learner import FaceRecognitionLearner

recognizer = FaceRecognitionLearner(backbone='ir_50', mode='backbone_only', device='cuda') # Initialize the recognizer
recognizer.load('./temp/saved_models') # Load the pretrained backbone
recognizer.fit_reference(path='./data/aligned_imgs', save_path='./temp/camera_demo') # Create/Load the reference database

# load face detector model

base_dir = os.path.dirname(__file__)
prototxt_path = './demo/model_data/deploy.prototxt'
caffemodel_path = './demo/model_data/weights.caffemodel'
modelcv = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    modelcv.setInput(blob)
    detections = modelcv.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # inference only the part of the image containing a face to the face recognition module
            face_img = frame[startY:endY, startX:endX]
            if face_img.any():
                text = recognizer.infer(face_img)
                if text is None:
                    text = 'Not Found'
                    col = (0, 0, 255)
                else:
                    col = (0, 255, 0)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              col, 2)
                cv2.putText(frame, text , (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
```


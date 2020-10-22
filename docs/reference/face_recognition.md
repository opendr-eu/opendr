## face_recognition_learner Module

The *face_recognition_learner* module contains the FaceRecognition class, which inherits from the abstract method *Learner*.

### Class FaceRecognition
Bases: `engine.learners.Learner`

FaceRecognition class is OpenDR's implementation for training and using a model on the face recognition task.

The [FaceRecognition](#src.perception.face_recognition.FaceRecognition) class accepts the following arguments:

|Parameters:| | 
|:---|:-------------|
| |**lr: *float, default=0.1*** <br /> &nbsp; &nbsp; &nbsp;Specifies the initial learning rate to be used during training.|
| |**iters: *int, default=120*** <br /> &nbsp; &nbsp; &nbsp;Specifies the number of iterations the training should be run for.| 
| |**batch_size: *int, default=32*** <br /> &nbsp; &nbsp; &nbsp;Specifies the batch size to be used during training.|
| |**optimizer: *{'sgd'}, default='sgd'*** <br /> &nbsp; &nbsp; &nbsp;Specifies the optimizer to be used during training. Currently supports 'sgd' (stochastic gradient decent).|
| |**device: *{'cpu', 'cuda'}, default='cuda'*** <br /> &nbsp; &nbsp; &nbsp;Specifies the device to be used. |
| |**threshold: *float, default=0.0*** <br /> &nbsp; &nbsp; &nbsp;Specifies the threshold to whether accept or discard a sample as a match or not.|
| |**backbone: *{'resnet_50, 'resnet_101', 'resnet_152', 'ir_50', 'ir_101', 'ir_152', 'ir_se_50', 'ir_se_101', 'ir_se_152', 'mobilefacenet'}, default='ir_50'*** <br /> &nbsp; &nbsp; &nbsp;Specifies the backbone architecture.|
| |**network_head: *{'arcface, 'cosface', 'sphereface', 'am_softmax', 'classifier'}, default='arcface'*** <br /> &nbsp; &nbsp; &nbsp;Specifies the head architecture.|
| |**loss: *{'focal','softmax'}, default='focal'*** <br /> &nbsp; &nbsp; &nbsp;Specifies the loss to be used during training.|
| |**temp_path: *str, default='./temp'*** <br /> &nbsp; &nbsp; &nbsp;Specifies a temporary path for checkpoints to be saved in.|
| |**mode: *{'backbone_only, 'head_only', 'full'}, default='backbone_only'*** <br /> &nbsp; &nbsp; &nbsp; The module supports three modes:<br /> <br /> <ul><li>'backbone_only': used for inference. Only the backbone architecture is used, and inference is done through retrieval.<br /><br /></li><li>'head_only': used for training. Only the head architecture will be trained, using a pretrained backbone model.<br /><br /></li><li>'full': used to train a combined model (backbone + head) or to inference with a classifier head.<br /></li></ul>|
| |**checkpoint_after_iter: *int, default=0*** <br /> &nbsp; &nbsp; &nbsp;Specifies after how many training iterations a checkpoint should be saved. If it is set to 0 no checkpoints will be saved.|  
| |**checkpoint_load_iter: *int, default=0*** <br /> &nbsp; &nbsp; &nbsp;Specifies which checkpoint should be loaded. If it is set to 0, no checkpoints will be loaded.|    
| |**val_after: *int, default=0*** <br /> &nbsp; &nbsp; &nbsp;Specifies after how many training iterations a validation should be run.|          
| |**input_size: *list, default=[112, 112]*** <br /> &nbsp; &nbsp; &nbsp;Specifies the input size of the images.|         
| |**rgb_mean: *list, default=[0.5, 0.5, 0.5]*** <br /> &nbsp; &nbsp; &nbsp;Specifies the mean values with which the input should be normalized.|   
| |**std_mean: *list, default=[0.5, 0.5, 0.5]*** <br /> &nbsp; &nbsp; &nbsp;Specifies the standard deviation values with which the input should be normalized.|   
| |**embedding_size: *int, default=512*** <br /> &nbsp; &nbsp; &nbsp;Specifies the size of the backbone's output.|
| |**weight_decay: *float, default=5e-4*** <br /> &nbsp; &nbsp; &nbsp;Specifies the rate the weights should be decayed by the optimizer during training.| 
| |**momentum: *float, default=0.9*** <br /> &nbsp; &nbsp; &nbsp;Specifies the momentum used by the optimized during training.| 
| |**drop_last: *bool, default=True*** <br /> &nbsp; &nbsp; &nbsp;Specifies whether the last batch should be dropped or not if it is not complete.|
| |**stages: *list, default=[8, 16, 24]*** <br /> &nbsp; &nbsp; &nbsp;Specifies on which iterations the learning rate should be adjusted.|
| |**pin_memory: *bool, default=True*** <br /> &nbsp; &nbsp; &nbsp;Specifies if pinned memory should be used by the Dataloader.|
| |**num_workers: *int, default=4*** <br /> &nbsp; &nbsp; &nbsp;Specifies the number of workers to be used by the Dataloader.|
   

The [FaceRecognition](#src.perception.face_recognition.FaceRecognition) class has the following public methods:

---

|fit(dataset, val_dataset, logging_path, silent, verbose)|
|:---|
This method is used for training the algorithm on a train dataset and validating on a val dataset.

| | | 
|:---|:-------------|
|Parameters:  | **dataset: *object*** <br /> Object that holds the training dataset|
| | **val_dataset: *object*** <br /> Object that holds the validation dataset|
| | **logging_path: *str, default=''*** <br />  path to save tensorboard log files. If set to None or ‘’, tensorboard logging is disabled|
| | **silent: *bool, default=False*** <br /> if set to True, disables printing training progress reports to STDOUT|
| | **verbose: *bool, default=True*** <br /> if set to True, enables the maximum logging verbosity|
|**Returns**: | ***dict***<br />Returns stats regarding training and validation | 

**Notes**

Train dataset should be of type ImageFolder. Images should be placed in a defined structure like:
- imgs
    - ID1
    - ID2
    - ID3
    - ...

When training a classifier head, the val_dataset should also be of type *'imagefolder'* of the same classes as the training dataset since the model is trained to predict specific classes.

---

|fit_reference(path, save_path)|
|:---|
This method is used to create a reference database to be used in inference when mode='backbone_only'.

| | | 
|:---|:-------------|
|Parameters:  | **path: *str, default=None*** <br />   path containing the reference images. If a reference database was already created can be left blank.|
| | **save_path: *bool, default=False*** <br /> path to save/load the .pkl reference file|

**Notes**

Reference images should be placed in a defined structure like:
- imgs
    - ID1
    - ID2
    - ID3
    - ...
    
When calling infer the method returns the name of the sub-folder, e.g. ID1

---

|infer(img)|
|:---|
This method is used to perform face recognition on an image.

|| | 
|:---|:-------------|
| Parameters: | **img: *object*** <br /> object of type engine.data.Image |
|**Returns** ('backbone_only' mode): |***str*** <br /> Returns the name of the sub-folder the best match in the reference database was in. |
|**Returns** ('classifier' head): |***int*** <br /> Returns the class ID.|  

---

|eval(dataset)|
|:---|
This method is used to perform face recognition on an image.

|| | 
|:---|:-------------|
| Parameters: | **dataset: *object*** <br /> object of type engine.datasets.ExternalDataset. Supports: 
| | <ul><li>'lfw' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)) <br /></li><li>'cfp_ff' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)) <br /></li><li>'cfp_fp' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)) <br /></li><li>'agedb_30' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)) <br /></li><li>'vgg2_fp' (as provided by [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch)) <br /></li><li>'ImageFolder'<br /></li></ul>|
|**Returns**: | ***dict***<br />Returns stats regarding evaluation | 

---

|save(path)|
|:---|
This method is used to save a trained model.

|| | 
|:---|:-------------|
| Parameters: | **path: *str*** <br /> path to save the model.|

---

|load(path)|
|:---|
This method is used to load a trained model.

|| | 
|:---|:-------------|
| Parameters: | **path: *str*** <br /> path of the model to be loaded.|

---

|load_from_onnx(path)|
|:---|
This method is used to load an optimized to onnx trained model.

|| | 
|:---|:-------------|
| Parameters: | **path: *str*** <br /> path of the onnx model to be loaded.

---

|optimize(path, do_constant_folding)|
|:---|
This method is used to optimize and export a trained model to onnx format.

|| | 
|:---|:-------------|
| Parameters: | **path: *str*** <br /> path of the onnx model to be loaded.|

---

**Examples**

---

*fit Example*
>```python
>from OpenDR.perception.face_recognition.face_recognition_learner import FaceRecognition
>from OpenDR.engine.datasets import ExternalDataset
>recognizer = FaceRecognition(backbone='ir_50', mode='full', network_head='arcface', iters=120, lr=0.1, checkpoint_after_iter=10, checkpoint_load_iter=0, device='cuda', val_after=10)
>train = ExternalDataset(path='./data', dataset_type='imagefolder')
>evaluation = ExternalDataset(path='./data', dataset_type='vgg2_fp')
>recognizer.fit(dataset=train, val_dataset=evaluation)
>recognizer.save('./temp')
>```

---

*infer example - backbone_only mode*
>```python
>import cv2
>from OpenDR.perception.face_recognition.face_recognition_learner import FaceRecognition
>recognizer = FaceRecognition(backbone='ir_50', mode='backbone_only', device='cuda')
>recognizer.load('./temp')
>recognizer.fit_reference(path='./data/imgs', save_path='./temp/demo')
>img = cv2.imread('test.jpg')
>result = recognizer.infer(img)
>print(result)
>```


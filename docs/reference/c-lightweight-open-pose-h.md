## C_API: lightweight_open_pose.h


The *lightweight_open_pose.h* header provides function definitions that can be used for accessing the OpenDR Lightweight OpenPose, pose estimation tool.

### Struct *OpenPoseModelT*
```C
struct OpenPoseModel {
 ...
};
typedef struct OpenPoseModel OpenPoseModelT;
```
The *OpenPoseModelT* structure keeps all the necessary information that are required by the OpenDR pose estimation tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *loadOpenPoseModel()*
```C
void loadOpenPoseModel(const char *modelPath, OpenPoseModelT *model);
```
 Loads a pose estimation model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freeOpenPoseModel()*
```C
void freeOpenPoseModel(OpenPoseModelT *model);
```
Releases the memory allocated for a pose estimation model (*model*).


### Function *forwardOpenPose()*
```C
void forwardOpenPose(OpenPoseModelT *model, OpenDRTensorT *tensor, OpenDRTensorVectorT *vector);
```
This function performs a forward pass using a pose estimation model (*model*) and an input tensor (*tensor*).
The function saves the output to an OpenDR vector of tensors structure (*vector*).


### Function *initRandomOpenDRTensorOp()*
```C
void initRandomOpenDRTensorOp(OpenDRTensorT *tensor, OpenPoseModelT *model);
```
This is used to initialize a random OpenDR tensor structure (*tensor*) with the appropriate dimensions for the pose estimation model (*model*).
The (*model*) keeps all the necessary information.

### Function *initOpenDRTensorFromImgOp()*
```C
void initOpenDRTensorFromImgOp(OpenDRImageT *image, OpenDRTensorT *tensor, OpenPoseModelT *model);
```
This is used to initialize an OpenDR tensor structure (*tensor*) with the data from an OpenDR image (*image*) for the pose estimation model (*model*).
The (*model*) keeps all the necessary information.

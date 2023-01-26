## C_API: lightweight_open_pose.h


The *lightweight_open_pose.h* header provides function definitions that can be used for accessing the OpenDR lightweight open pose, pose estimation tool.

### Struct *OpenPoseModelT*
```C
struct OpenPoseModel {
 ...
};
typedef struct OpenPoseModel OpenPoseModelT;
```
The *OpenPoseModelT* structure keeps all the necessary information that are required by the OpenDR lightweight open pose tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *loadOpenPoseModel()*
```C
void loadOpenPoseModel(const char *modelPath, OpenPoseModelT *model);
```
 Loads a lightweight open pose, pose estimation model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freeOpenPoseModel()*
```C
void freeOpenPoseModel(OpenPoseModelT *model);
```
Releases the memory allocated for a pose estimation lightweight open pose model (*model*).


### Function *forwardOpenPose()*
```C
void forwardOpenPose(OpenPoseModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);
```
This function perform forward pass using a pose estimation, lightweight open pose model (*model*) and an input tensor (*tensor*).
The function saves the output to an OpenDR vector of tensors structure (*vector*).


### Function *initRandomOpendrTensorOp()*
```C
void initRandomOpendrTensorOp(OpendrTensorT *tensor, OpenPoseModelT *model);
```
This is used to initialize a random OpenDR tensor structure (*tensor*) with the appropriate dimensions for the pose estimation lightweight open pose model (*model*).
The (*model*) keeps all the necessary information.

### Function *initOpendrTensorFromImgOp()*
```C
void initOpendrTensorFromImgOp(OpendrImageT *image, OpendrTensorT *tensor, OpenPoseModelT *model);
```
This is used to initialize an OpenDR tensor structure (*tensor*) with the data from an OpenDR image (*image*) for the lightweight open pose model (*model*).
The (*model*) keeps all the necessary information.

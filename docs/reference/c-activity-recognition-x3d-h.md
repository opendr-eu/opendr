## C_API: activity_recognition_2d_x3d.h


The *activity_recognition_2d_x3d.h* header provides function definitions that can be used for accessing the OpenDR activity recognition X3D tool.

### Struct *X3dModelT*
```C
struct X3dModel {
  ...
};
typedef struct X3dModel X3dModelT;
```
The *X3dModelT* structure keeps all the necessary information that are required by the OpenDR activity recognition X3D tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *loadX3dModel()*
```C
void loadX3dModel(const char *modelPath, char *mode, X3dModelT *model);
```
 Loads a X3D activity recognition model saved in the local filesystem (*modelPath*) in OpenDR format, with the expected hyperparameters of (*mode*).
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freeX3dModel()*
```C
void freeX3dModel(X3dModelT *model);
```
Releases the memory allocated for an activity recognition X3D model (*model*).


### Function *forwardX3d()*
```C
void forwardX3d(X3dModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);
```
This function performs a forward pass using an activity recognition X3D model (*model*) and an input tensor (*tensor*).
The function saves the output to an OpenDR vector of tensors structure (*vector*).


### Function *initRandomOpendrTensorX3d()*
```C
void initRandomOpendrTensorX3d(OpendrTensorT *tensor, X3dModelT *model);
```
This is used to initialize a random OpenDR tensor structure (*tensor*) with the appropriate dimensions for the activity recognition X3D model (*model*).
The (*model*) keeps all the necessary information.


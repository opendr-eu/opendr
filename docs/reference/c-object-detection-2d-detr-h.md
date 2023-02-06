## C_API: object_detection_2d_detr.h


The *object_detection_2d_detr.h* header provides function definitions that can be used for accessing the OpenDR object detection 2D DETR tool.

### Struct *DetrModelT*
```C
struct DetrModel {
  ...
};
typedef struct DetrModel DetrModelT;
```
The *DetrModelT* structure keeps all the necessary information that are required by the OpenDR object detection 2D DETR tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *loadDetrModel()*
```C
void loadDetrModel(const char *modelPath, DetrModelT *model);
```
 Loads a DETR object detection model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freeDetrModel()*
```C
void freeDetrModel(DetrModelT *model);
```
Releases the memory allocated for an object detection DETR model (*model*).


### Function *forwardDetr()*
```C
void forwardDetr(DetrModelT *model, OpenDRTensorT *tensor, OpenDRTensorVectorT *vector);
```
This function performs a forward pass using an object detection 2D DETR model (*model*) and an input tensor (*tensor*).
The function saves the output to an OpenDR vector of tensors structure (*vector*).


### Function *initRandomOpenDRTensorDetr()*
```C
void initRandomOpenDRTensorDetr(OpenDRTensorT *tensor, DetrModelT *model);
```
This is used to initialize a random OpenDR tensor structure (*tensor*) with the appropriate dimensions for the object detection DETR model (*model*).
The (*model*) keeps all the necessary information.


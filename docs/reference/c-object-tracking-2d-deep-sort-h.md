## C_API: object_tracking_2d_deep_sort.h


The *object_tracking_2d_deep_sort.h* header provides function definitions that can be used for accessing the OpenDR object tracking 2d deep sort tool.

### Struct *DeepSortModelT*
```C
struct DeepSortModel {
  ...
};
typedef struct DeepSortModel DeepSortModelT;
```
The *DeepSortModelT* structure keeps all the necessary information that are required by the OpenDR object tracking 2d deep sort tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *loadDeepSortModel()*
```C
void loadDeepSortModel(const char *modelPath, DeepSortModelT *model);
```
 Loads a deep sort object tracking model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freeDeepSortModel()*
```C
void freeDeepSortModel(DeepSortModelT *model);
```
Releases the memory allocated for an object tracking 2d deep sort model (*model*).


### Function *forwardDeepSort()*
```C
void forwardDeepSort(DeepSortModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);
```
This function perform forward pass using an object tracking 2d deep sort model (*model*) and an input tensor (*tensor*).
The function saves the output to an OpenDR vector of tensors structure (*vector*).


### Function *initRandomOpendrTensorDs()*
```C
void initRandomOpendrTensorDs(OpendrTensorT *tensor, DeepSortModelT *model);
```
This is used to initialize a random OpenDR tensor structure (*tensor*) with the appropriate dimensions for the object tracking deep sort model (*model*).
The (*model*) keeps all the necessary information.


## C_API: skeleton_based_action_recognition_pst.h


The *skeleton_based_action_recognition_pst.h* header provides function definitions that can be used for accessing the OpenDR Skeleton-based Action Recognition Progressive Spatiotemporal GCN tool.

### Struct *PstModelT*
```C
struct PstModel {
  ...
};
typedef struct PstModel PstModelT;
```
The *PstModelT* structure keeps all the necessary information that are required by the OpenDR Skeleton-based Action Recognition Progressive Spatiotemporal GCN tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *loadPstModel()*
```C
void loadPstModel(const char *modelPath, PstModelT *model);
```
 Loads a Progressive Spatiotemporal GCN Skeleton-based Action Recognition model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freePstModel()*
```C
void freePstModel(PstModelT *model);
```
Releases the memory allocated for a Skeleton-based Action Recognition Progressive Spatiotemporal GCN model (*model*).


### Function *forwardPst()*
```C
void forwardPst(PstModelT *model, OpenDRTensorT *tensor, OpenDRTensorVectorT *vector);
```
This function performs a forward pass using a skeleton-based Action Recognition model (*model*) and an input tensor (*tensor*).
The function saves the output to an OpenDR vector of tensors structure (*vector*).


### Function *initRandomOpenDRTensorPst()*
```C
void initRandomOpenDRTensorPst(OpenDRTensorT *tensor, PstModelT *model);
```
This is used to initialize a random OpenDR tensor structure (*tensor*) with the appropriate dimensions for the Skeleton-based Action Recognition Progressive Spatiotemporal GCN model (*model*).
The (*model*) keeps all the necessary information.


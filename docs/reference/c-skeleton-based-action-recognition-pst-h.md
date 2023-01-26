## C_API: skeleton_based_action_recognition_pst.h


The *skeleton_based_action_recognition_pst.h* header provides function definitions that can be used for accessing the OpenDR skeleton based action recognition progressive spatiotemporal gcn tool.

### Struct *PstModelT*
```C
struct PstModel {
  ...
};
typedef struct PstModel PstModelT;
```
The *PstModelT* structure keeps all the necessary information that are required by the OpenDR skeleton based action recognition progressive spatiotemporal gcn tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *loadPstModel()*
```C
void loadPstModel(const char *modelPath, PstModelT *model);
```
 Loads a progressive spatiotemporal gcn skeleton based action recognition model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freePstModel()*
```C
void freePstModel(PstModelT *model);
```
Releases the memory allocated for a skeleton based action recognition progressive spatiotemporal gcn model (*model*).


### Function *forwardPst()*
```C
void forwardPst(PstModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);
```
This function perform forward pass using a skeleton based action recognition progressive spatiotemporal gcn model (*model*) and an input tensor (*tensor*).
The function saves the output to an OpenDR vector of tensors structure (*vector*).


### Function *initRandomOpendrTensorPst()*
```C
void initRandomOpendrTensorPst(OpendrTensorT *tensor, PstModelT *model);
```
This is used to initialize a random OpenDR tensor structure (*tensor*) with the appropriate dimensions for the skeleton based action recognition progressive spatiotemporal gcn model (*model*).
The (*model*) keeps all the necessary information.


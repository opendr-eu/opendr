## C_API: skeleton_based_action_recognition_pst.h


The *skeleton_based_action_recognition_pst.h* header provides function definitions that can be used for accessing the OpenDR skeleton based action recognition progressive spatiotemporal gcn tool.

### Struct *pst_model_t*
```C
struct pst_model {
  ...
};
typedef struct pst_model pst_model_t;
```
The *pst_model_t* structure keeps all the necessary information that are required by the OpenDR skeleton based action recognition progressive spatiotemporal gcn tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *load_pst_model()*
```C
void load_pst_model(char *modelPath, pst_model_t *model);
```
 Loads a progressive spatiotemporal gcn skeleton based action recognition model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a (*device*) Jit network for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *free_pst_model()*
```C
void free_pst_model(pst_model_t *model);
```
Releases the memory allocated for a skeleton based action recognition progressive spatiotemporal gcn model (*model*).


### Function *forward_pst()*
```C
void forward_pst(pst_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector);
```
This function perform forward pass using a skeleton based action recognition progressive spatiotemporal gcn model (*model*) and an input tensor (*inputTensorValues*).
The function returns an OpenDR tensor vector structure with the forward pass results.


### Function *init_random_opendr_tensor_pst()*
```C
void init_random_opendr_tensor_pst(opendr_tensor_t *inputTensorValues, pst_model_t *model);
```
This is used to initialize a random OpenDR tensor structure (*inputTensorValues*) with the appropriate dimensions for the skeleton based action recognition progressive spatiotemporal gcn model (*model*).
The (*model*) keeps all the necessary information.


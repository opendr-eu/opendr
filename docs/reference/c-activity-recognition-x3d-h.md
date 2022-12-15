## C_API: activity_recognition_2d_x3d_jit.h


The *activity_recognition_2d_x3d.h* header provides function definitions that can be used for accessing the OpenDR activity recognition 2d x3d tool.

### Struct *x3d_model_t*
```C
struct x3d_model {
  ...
};
typedef struct x3d_model x3d_model_t;
```
The *x3d_model_t* structure keeps all the necessary information that are required by the OpenDR activity recognition 2d x3d tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *load_x3d_model()*
```C
void load_x3d_model(char *modelPath, x3d_model_t *model);
```
 Loads a x3d activity recognition model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a (*device*) Jit network for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *free_x3d_model()*
```C
void free_x3d_model(x3d_model_t *model);
```
Releases the memory allocated for an activity recognition 2d x3d model (*model*).


### Function *forward_x3d()*
```C
void forward_x3d(x3d_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector);
```
This function perform forward pass using an activity recognition 2d x3d model (*model*) and an input tensor (*inputTensorValues*).
The function returns an OpenDR tensor vector structure with the forward pass results.


### Function *init_random_opendr_tensor_x3d()*
```C
void init_random_opendr_tensor_x3d(opendr_tensor_t *inputTensorValues, x3d_model_t *model);
```
This is used to initialize a random OpenDR tensor structure (*inputTensorValues*) with the appropriate dimensions for the activity recognition x3d model (*model*).
The (*model*) keeps all the necessary information.


## C_API: object_tracking_2d_deep_sort.h


The *object_tracking_2d_deep_sort.h* header provides function definitions that can be used for accessing the OpenDR object tracking 2d deep sort tool.

### Struct *deep_sort_model_t*
```C
struct deep_sort_model {
  ...
};
typedef struct deep_sort_model deep_sort_model_t;
```
The *deep_sort_model_t* structure keeps all the necessary information that are required by the OpenDR object tracking 2d deep sort tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *load_deep_sort_model()*
```C
void load_deep_sort_model(char *modelPath, deep_sort_model_t *model);
```
 Loads a deep sort object tracking model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *free_deep_sort_model()*
```C
void free_deep_sort_model(deep_sort_model_t *model);
```
Releases the memory allocated for an object tracking 2d deep sort model (*model*).


### Function *forward_deep_sort()*
```C
void forward_deep_sort(deep_sort_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector);
```
This function perform forward pass using an object tracking 2d deep sort model (*model*) and an input tensor (*inputTensorValues*).
The function returns an OpenDR tensor vector structure with the forward pass results.


### Function *init_random_opendr_tensor_ds()*
```C
void init_random_opendr_tensor_ds(opendr_tensor_t *inputTensorValues, deep_sort_model_t *model);
```
This is used to initialize a random OpenDR tensor structure (*inputTensorValues*) with the appropriate dimensions for the object tracking deep sort model (*model*).
The (*model*) keeps all the necessary information.


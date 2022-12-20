## C_API: object_detection_2d_detr.h


The *object_detection_2d_detr.h* header provides function definitions that can be used for accessing the OpenDR object detection 2d detr tool.

### Struct *detr_model_t*
```C
struct detr_model {
  ...
};
typedef struct detr_model detr_model_t;
```
The *detr_model_t* structure keeps all the necessary information that are required by the OpenDR object detection 2d detr tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *load_detr_model()*
```C
void load_detr_model(char *modelPath, detr_model_t *model);
```
 Loads a detr object detection model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *free_detr_model()*
```C
void free_detr_model(detr_model_t *model);
```
Releases the memory allocated for an object detection 2d detr model (*model*).


### Function *forward_detr()*
```C
void forward_detr(detr_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector);
```
This function perform forward pass using an object detection 2d detr model (*model*) and an input tensor (*inputTensorValues*).
The function returns an OpenDR tensor vector structure with the forward pass results.


### Function *init_random_opendr_tensor_detr()*
```C
void init_random_opendr_tensor_detr(opendr_tensor_t *inputTensorValues, detr_model_t *model);
```
This is used to initialize a random OpenDR tensor structure (*inputTensorValues*) with the appropriate dimensions for the object detection detr model (*model*).
The (*model*) keeps all the necessary information.


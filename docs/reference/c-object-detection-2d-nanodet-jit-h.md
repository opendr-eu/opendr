## C_API: object_detection_2d_nanodet_jit.h


The *object_detection_2d_nanodet_jit.h* header provides function definitions that can be used for accessing the OpenDR object detection 2d nanodet tool.

### Struct *nanodet_model_t*
```C
struct nanodet_model {
  ...
};
typedef struct nanodet_model nanodet_model_t;
```
The *nanodet_model_t* structure keeps all the necessary information that are required by the OpenDR object detection 2d nanodet tool (e.g., model weights, normalization information, etc.).


### Function *load_nanodet_model()*
```C
void load_nanodet_model(char *modelPath, char *device, int height, int width, float scoreThreshold, nanodet_model_t *model);
```
 Loads a nanodet object detection model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a (*device*) Jit network for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *free_nanodet_model()*
```C
void free_nanodet_model(nanodet_model_t *model);
```
Releases the memory allocated for an object detection 2d nanodet model (*model*).


### Function *infer_nanodet()*
```C
opendr_detection_vector_target_t infer_nanodet(nanodet_model_t *model, opendr_image_t *image);
```
This function perform inference using an object detection 2d nanodet model (*model*) and an input image (*image*).
The function returns an OpenDR detection vector structure with the inference results.


### Function *draw_bboxes()*
```C
void draw_bboxes(opendr_image_t *image, nanodet_model_t *model, opendr_detection_vector_target_t *detectionsVector);
```
This function draws the given detections (*detectionsVector*) into the input image (*image*) and then show it in screen.
The (*model*) keeps all the necessary information.


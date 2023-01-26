## C_API: object_detection_2d_nanodet_jit.h


The *object_detection_2d_nanodet_jit.h* header provides function definitions that can be used for accessing the OpenDR object detection 2D Nanodet tool.

### Struct *NanodetModelT*
```C

struct NanodetModel {
  ...
};
typedef struct NanodetModel NanodetModelT;
```
The *NanodetModelT* structure keeps all the necessary information that are required by the OpenDR object detection 2D Nanodet tool (e.g., model weights, normalization information, etc.).


### Function *loadNanodetModel()*
```C
void loadNanodetModel(char *modelPath, char *device, int height, int width, float scoreThreshold, NanodetModelT *model);
```
Loads a Nanodet object detection model saved in the local filesystem (*modelPath*) in OpenDR format.
This function also initializes a (*device*) JIT network for performing inference using this model.
The pre-trained models should follow the OpenDR conventions.
The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.

### Function *freeNanodetModel()*
```C
void freeNanodetModel(NanodetModelT *model);
```
Releases the memory allocated for an object detection 2D Nanodet model (*model*).


### Function *inferNanodet()*
```C
OpendrDetectionVectorTargetT inferNanodet(NanodetModelT *model, OpendrImageT *image);
```
This function performs inference using an object detection 2D Nanodet model (*model*) and an input image (*image*).
The function returns an OpenDR vector of detections structure with the inference results.


### Function *drawBboxes()*
```C
void drawBboxes(OpendrImageT *image, NanodetModelT *model, OpendrDetectionVectorTargetT *vector);
```
This function draws the given detections (*vector*) onto the input image (*image*) and then shows the image on screen.
The (*model*) keeps all the necessary information.


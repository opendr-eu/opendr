## C_API: opendr_utils.h


The *opendr_utils.h* header provides function definitions of OpenDR helpers (e.g., for creating OpenDR images).

### Function *jsonGetKeyString()*
```C
const char* jsonGetKeyString(const char *json, const char *key, const int index);
```
The *jsonGetKeyString()* function reads a JSON string from the pointer (*json*) and returns tha value of a key with pointer (*key*) as string.
If the value is an array it will return only the (*index*) value of the array.
If fails returns (*""*).

### Function *jsonGetKeyFloat()*
```C
float jsonGetKeyFloat(const char *json, const char *key, const int index);
```
The *jsonGetKeyFloat()* function reads a JSON string from the pointer (*json*) and returns tha value of a key with pointer (*key*) as float.
If the value is an array it will return only the (*index*) value of the array.
If fails returns (*0.0f*).

### Function *jsonGetKeyFromInferenceParams()*
```C
float jsonGetKeyFromInferenceParams(const char *json, const char *key, const int index);
```
The *jsonGetKeyFromInferenceParams()* function reads a JSON string from the pointer (*json*) and returns tha value of a key with pointer (*key*) in inference_params section as float.
If the value is an array it will return only the (*index*) value of the array.
If fails returns (*0.0f*).

---

### Function *loadImage()*
```C
void loadImage(const char *path, OpendrImageT *image);
```
The *loadImage()* function loads an image from the local file system (*path*) into an OpenDR image data type.
A pointer (*image*) to an OpenDR *OpendrImageT* should be provided.
This function allocates memory during each function call, so be sure to use the *freeImage()* function to release the allocated resources, when the corresponding image is no longer needed.

### Function *freeImage()*
```C
void freeImage(OpendrImageT *image);
```
The *freeImage()* function releases the memory allocated for an OpenDR image structure (*image*).
A pointer (*image*) to an OpenDR *OpendrImageT* should be provided.

---

### Function *initDetectionsVector()*
```C
void initDetectionsVector(OpendrDetectionVectorTargetT *detectionVector);
```
The *initDetectionsVector()* function initializes the data of an OpenDR detection vector structure (*detectionVector*) with zero values.
A pointer (*detectionVector*) to an OpenDR *DetectionVectorTargetT* should be provided.

### Function *loadDetectionsVector()*
```C
void loadDetectionsVector(OpendrDetectionVectorTargetT *detectionVector, OpendrDetectionTargetT *detection,
                          int vectorSize);
```
The *loadDetectionsVector()* function stores OpenDR detection target structures in the memory allocated for multiple OpenDR detections structures (*detection*).
A pointer (*detectionVector*) to an OpenDR *OpendrDetectionVectorTargetT* should be provided.

### Function *freeDetectionsVector()*
```C
void freeDetectionsVector(OpendrDetectionVectorTargetT *detectionVector);
```
The *freeDetectionsVector()* function releases the memory allocated for an OpenDR detection vector structure (*detectionVector*).
A pointer (*detectionVector*) to an OpenDR *OpendrDetectionVectorTargetT* should be provided.

## C_API: opendr_utils.h


The *opendr_utils.h* header provides function definitions of OpenDR helpers (e.g., for creating OpenDR images).

### Function *jsonGetStringFromKey()*
```C
const char *jsonGetStringFromKey(const char *json, const char *key, const int index);
```
The *jsonGetStringFromKey()* function reads a JSON string from the pointer (*json*) and returns the value of a key with pointer (*key*) as string.
If the value is an array it will return only the (*index*) value of the array.
If it fails it returns ("").

### Function *jsonGetFloatFromKey()*
```C
float jsonGetFloatFromKey(const char *json, const char *key, const int index);
```
The *jsonGetFloatFromKey()* function reads a JSON string from the pointer (*json*) and returns the value of a key with pointer (*key*) as float.
If the value is an array it will return only the (*index*) value of the array.
If it fails it returns (*0.0f*).

### Function *jsonGetBoolFromKey()*
```C
int jsonGetBoolFromKey(const char *json, const char *key, const int index);
```
The *jsonGetBoolFromKey()* function reads a JSON string from the pointer (*json*) and returns an integer based on the boolean value of a key with pointer (*key*) as integer.
If the value is an array it will return only the (*index*) value of the array.
If the value is true it will return 0, else it will return 1.
If it fails it returns (*-1*).

### Function *jsonGetStringFromKeyInInferenceParams()*
```C
const char *jsonGetStringFromKeyInInferenceParams(const char *json, const char *key, const int index);
```
The *jsonGetStringFromKeyInInferenceParams()* function reads a JSON string from the pointer (*json*) and returns the value of a key with pointer (*key*) in inference_params section as string.
If the value is an array it will return only the (*index*) value of the array.
If it fails it returns ("").

### Function *jsonGetFloatFromKeyInInferenceParams()*
```C
const char *jsonGetFloatFromKeyInInferenceParams(const char *json, const char *key, const int index);
```
The *jsonGetFloatFromKeyInInferenceParams()* function reads a JSON string from the pointer (*json*) and returns the value of a key with pointer (*key*) in inference_params section as float.
If the value is an array it will return only the (*index*) value of the array.
If it fails it returns (*0.0f*).

### Function *jsonGetBoolFromKeyInInferenceParams()*
```C
int jsonGetBoolFromKeyInInferenceParams(const char *json, const char *key, const int index);
```
The *jsonGetBoolFromKeyInInferenceParams()* function reads a JSON string from the pointer (*json*) and returns an integer based on the boolean value of a key with pointer (*key*) in inference_params section as integer.
If the value is an array it will return only the (*index*) value of the array.
If the value is true it will return 0, else it will return 1.
If it fails it returns (*-1*).

---

### Function *loadImage()*
```C
void loadImage(const char *path, OpenDRImageT *image);
```
The *loadImage()* function loads an image from the local file system (*path*) into an OpenDR image data type.
A pointer (*image*) to an OpenDR *OpenDRImageT* should be provided.
This function allocates memory during each function call, so be sure to use the *freeImage()* function to release the allocated resources, when the corresponding image is no longer needed.

### Function *freeImage()*
```C
void freeImage(OpenDRImageT *image);
```
The *freeImage()* function releases the memory allocated for an OpenDR image structure (*image*).
A pointer (*image*) to an OpenDR *OpenDRImageT* should be provided.

---

### Function *initDetectionsVector()*
```C
void initDetectionsVector(OpenDRDetectionVectorTargetT *vector);
```
The *initDetectionsVector()* function initializes the data of an OpenDR detection vector structure (*vector*) with zero values.
A pointer (*vector*) to an OpenDR *DetectionVectorTargetT* should be provided.

### Function *loadDetectionsVector()*
```C
void loadDetectionsVector(OpenDRDetectionVectorTargetT *vector, OpenDRDetectionTargetT *detectionPtr, int vectorSize);
```
The *loadDetectionsVector()* function stores OpenDR vector of detections structures in the memory allocated for multiple OpenDR detections structures (*detectionPtr*).
A pointer (*vector*) to an OpenDR *OpenDRDetectionVectorTargetT* should be provided.

### Function *freeDetectionsVector()*
```C
void freeDetectionsVector(OpenDRDetectionVectorTargetT *vector);
```
The *freeDetectionsVector()* function releases the memory allocated for an OpenDR vector of detections structure (*vector*).
A pointer (*vector*) to an OpenDR *OpenDRDetectionVectorTargetT* should be provided.

---

### Function *initTensor()*
```C
void initTensor(OpenDRTensorT *tensor);
```
The *initTensor()* function initializes the data of an OpenDR tensor (*tensor*) with zero values.
A pointer (*tensor*) to an OpenDR *OpenDRTensorT* should be provided.

### Function *loadTensor()*
```C
void loadTensor(OpenDRTensorT *tensor, void *tensorData, int batchSize, int frames, int channels, int width, int height);
```
The *loadTensor()* function allows for storing OpenDR tensor structures in to the memory allocated into a pointer into the OpenDR tensor structure (*tensor*).
A pointer (*tensor*) to an OpenDR *OpenDRTensorT* along with the pointer into the memory (*tensorData*) and the (*batchSize*), (*frames*), (*channels*), (*width*) and (*height*) of tensor should be provided. 
All integers must have a minimum value of *1*.

### Function *freeTensor()*
```C
void freeTensor(OpenDRTensorT *tensor);
```
The *freeTensor()* function releases the memory allocated for an OpenDR tensor structure (*tensor*).
A pointer (*tensor*) to an OpenDR *OpenDRTensorT* should be provided.

### Function *initTensorVector()*
```C
void initTensorVector(OpenDRTensorVectorT *vector);
```
The *initTensorVector()* function initializes the data of an OpenDR vector of tensors (*vector*) with zero values.
A pointer (*vector*) to an OpenDR *OpenDRTensorVectorT* should be provided.

### Function *loadTensorVector()*
```C
void loadTensorVector(OpenDRTensorVectorT *vector, OpenDRTensorT *tensorPtr, int nTensors);
```
The *loadTensorVector()* function allows for storing multiple OpenDR tensor structures in memory allocated by the OpenDR vector of tensors structure (*vector*).
A pointer (*vector*) to an OpenDR *OpenDRTensorVectorT* along with the pointer into the memory of a vector or array of OpenDR tensors structure (*tensorPtr*) should be provided.
Moreover, the number of tensors (*nTensors*) should be included, and it must be equal or higher than *1*.

### Function *freeTensorVector()*
```C
void freeTensorVector(OpenDRTensorVectorT *vector);
```
The *freeTensorVector()* function releases the memory allocated for an OpenDR vector ot tensors structure (*vector*).
A pointer (*vector*) to an OpenDR *OpenDRTensorVectorT* should be provided.

### Function *iterTensorVector()*
```C
void iterTensorVector(OpenDRTensorT *tensor, OpenDRTensorVectorT *vector, int index);
```
The *iterTensorVector()* function is used to help the user to iterate the OpenDR vector of tensors.
A single OpenDR tensor (*tensor*) is loaded with the values of the indexed (*index*) tensor of the (*vector*).
A pointer (*vector*) to an OpenDR *OpenDRTensorVectorT* and an (*index*) along with a pointer (*tensor*) to an OpenDR *OpenDRTensorT* should be provided.
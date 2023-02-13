## C_API: face_recognition.h


The *face_recognition.h* header provides function definitions that can be used for accessing the OpenDR face recognition tool.

### Struct *FaceRecognitionModelT*
```C

struct FaceRecognitionModel {
...
};
typedef struct FaceRecognitionModel FaceRecognitionModelT;
```
The *FaceRecognitionModelT* structure keeps all the necessary information that are required by the OpenDR face recognition tool (e.g., model weights, normalization information, database for person recognition, ONNX session information, etc.).


### Function *loadFaceRecognitionModel()*
```C
void loadFaceRecognitionModel(const char *modelPath, FaceRecognitionModelT *model);
```
 Loads a face recognition model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *freeFaceRecognitionModel()*
```C
void freeFaceRecognitionModel(FaceRecognitionModelT *model);
```
Releases the memory allocated for a face recognition model (*model*).


### Function *inferFaceRecognition()*
```C
OpenDRCategoryTargetT inferFaceRecognition(FaceRecognitionModelT *model, OpenDRImageT *image);
```
This function performs inference using a face recognition model (*model*) and an input image (*image*).
The function returns an OpenDR category structure with the inference results.


### Function *decodeCategoryFaceRecognition()*
```C
void decodeCategoryFaceRecognition(FaceRecognitionModelT *model, OpenDRCategoryTargetT category, char *personName);
```
Returns the name of a recognized person by decoding the category ID into a string (this function uses the information from the built person database).


### Function *buildDatabaseFaceRecognition()*
```C
void buildDatabaseFaceRecognition(const char *databaseFolder, const char *outputPath, faceRecognitionModelT *model);
```
Build a face recognition database (containing images for persons to be recognized). 
This function expects the (*databaseFolder*) to have the same format as the main Python toolkit.
The function calculates the features of the person that are contained in the database and it stores it into a binary file that can be then loaded to perform inference (*outputPath*).
A loaded face recognition model should be provided (*model*), since this model will be used for the feature extraction process.

### Function *loadDatabaseFaceRecognition()*
```C
void loadDatabaseFaceRecognition(const char *databasePath, FaceRecognitionModelT *model);

```
Loads an already built database (*databasePath*) into a face recognition model (*model*).
After this step, the model can be used for performing inference. 


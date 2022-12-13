## C_API: face_recognition.h


The *face_recognition.h* header provides function definitions that can be used for accessing the OpenDR face recognition tool.

### Struct *face_recognition_model_t*
```C

struct face_recognition_model {
...
};
typedef struct face_recognition_model face_recognition_model_t;
```
The *face_recognition_model_t* structure keeps all the neccesary information that are required by the OpenDR face recognition tool (e.g., model weights, normalization information, database for person recognition, ONNX session information, etc.).


### Function *load_face_recognition_model()*
```C
void load_face_recognition_model(const char *model_path, face_recognition_model_t *model);
```
 Loads a face recognition model saved in the local filesystem (*model path*) in OpenDR format.
 This function also initializes a CPU-based ONNX session for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *free_face_recognition_model()*
```C
void free_face_recognition_model(face_recognition_model_t *model);
```
Releases the memory allocated for a face recognition model (*model*).


### Function *infer_face_recognition()*
```C
opendr_category_target_t infer_face_recognition(face_recognition_model_t *model, opendr_image_t *image);
```
This function perform inference using a face recognition model (*model*) and an input image (*image*).
The function returns an OpenDR category structure with the inference results.


### Function *decode_category_face_recognition()*
```C
void decode_category_face_recognition(face_recognition_model_t *model, opendr_category_target_t category, char *person_name);
```
Returns the name of a recognized person by decoding the category id into a string (this function uses the information from the built person database).


### Function *build_database_face_recognition()*
```C
void build_database_face_recognition(const char *database_folder, const char *output_path, face_recognition_model_t *model);
```
Build a face recognition database (containing images for persons to be recognized). 
This function expects the *database_folder* to have the same format as the main Python toolkit.
The function calculates the features of the person that are contained in the database and it stores it into a binary file that can be then loaded to perform inference (*output_path*).
A loaded face recongition model should be provided (*model*), since this model will be used for the feature extraction process.

### Function *load_database_face_recognition()*
```C
void load_database_face_recognition(const char *database_path, face_recognition_model_t *model);

```
Loads an already built database (*database_path) into a face recognition model (*model*).
After this step, the model can be used for performing inference. 


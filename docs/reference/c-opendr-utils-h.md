## C_API: opendr_utils.h


The *opendr_utils.h* header provides function definitions of OpenDR helpers (e.g., for creating OpenDR images).

### Function *json_get_key_string()*
```C
const char* json_get_key_string(const char *json, const char *key);
```
The *json_get_key_string()* function allows for reading a json file and return the value of a key.
A pointer (*json*) that have the json string and a pointer (*key*) with the wanted value is needed.

##

### Function *load_image()*
```C
void load_image(const char *path, opendr_image_t *image);
```
The *load_image()* function allows for reading an images from the local file system (*path*) into an OpenDR image data type.
A pointer (*image*) to an OpenDR *opendr_image_t* should be provided.
This function allocates memory during each function call, so be sure to use the *free_image()* function to release the allocated resources, when the corresponding image is no longer needed.

### Function *free_image()*
```C
void free_image(opendr_image_t *image);
```
The *free_image()* function releases the memory allocated for an OpenDR image structure (*image*).
A pointer (*image*) to an OpenDR *opendr_image_t* should be provided.

### Function *init_detections_vector()*
```C
void init_detections_vector(opendr_detection_vector_target_t *detection_vector);
```
The *init_detections_vector()* function initialize the data of an OpenDR detection vector structure (*detection_vector*) with zero values.
A pointer (*detection_vector*) to an OpenDR *detection_vector_target_t* should be provided.

### Function *load_detections_vector()*
```C
void load_detections_vector(opendr_detection_vector_target_t *detection_vector, opendr_detection_target_t *detection,
                            int vector_size);
```
The *load_detections_vector()* function allows for storing OpenDR detection target structures in to the memory allocated for multiple OpenDR detections structures (*detection*).
A pointer (*detection_vector*) to an OpenDR *opendr_detection_vector_target_t* should be provided.

### Function *free_detections_vector()*
```C
void free_detections_vector(opendr_detection_vector_target_t *detection_vector);
```
The *free_detections_vector()* function releases the memory allocated for an OpenDR detection vector structure (*detection_vector*).
A pointer (*detection_vector*) to an OpenDR *opendr_detection_vector_target_t* should be provided.

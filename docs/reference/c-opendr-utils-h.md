## C_API: opendr_utils.h


The *opendr_utils.h* header provides function definitions of OpenDR helpers (e.g., for creating OpenDR images).

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

### Function *initialize_detections_vector()*
```C
void initialize_detections_vector(opendr_detection_vector_target_t *detection_vector);
```
The *initialize_detections_vector()* function initialize the data of an OpenDR detection vector structure (*detection_vector*) with zero values.
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

### Function *initialize_tensor()*
```C
void initialize_tensor(opendr_tensor_t *opendr_tensor);
```
The *initialize_tensor()* function initialize the data of an OpenDR tensor (*opendr_tensor*) with zero values.
A pointer (*opendr_tensor*) to an OpenDR *opendr_tensor_t* should be provided.

### Function *load_tensor()*
```C
void load_tensor(opendr_tensor_t *opendr_tensor, void *tensor_data, int batch_size, int frames, int channels, int width,
                 int height);
```
The *load_tensor()* function allows for storing OpenDR tensor structures in to the memory allocated into a pointer into the OpenDR tensor structure (*opendr_tensor*).
A pointer (*opendr_tensor*) to an OpenDR *opendr_tensor_t* along with the pointer into the memory (*tensor_data*) and the (*batch_size*), (*frames*), (*channels*), (*width*) and (*height*) of tensor should be provided. 
All integers must have a minimum value of *1*.

### Function *free_tensor()*
```C
void free_tensor(opendr_tensor_t *opendr_tensor);
```
The *free_tensor()* function releases the memory allocated for an OpenDR tensor structure (*opendr_tensor*).
A pointer (*opendr_tensor*) to an OpenDR *opendr_tensor_t* should be provided.

### Function *initialize_tensor_vector()*
```C
void initialize_tensor_vector(opendr_tensor_vector_t *tensor_vector);
```
The *initialize_tensor_vector()* function initialize the data of an OpenDR tensor vector (*tensor_vector*) with zero values.
A pointer (*tensor_vector*) to an OpenDR *opendr_tensor_vector_t* should be provided.

### Function *load_tensor_vector()*
```C
void load_tensor_vector(opendr_tensor_vector_t *tensor_vector, opendr_tensor_t *tensor, int number_of_tensors);
```
The *load_tensor_vector()* function allows for storing multiple OpenDR tensor structures in to the memory allocated into pointers into the OpenDR tensor vector structure (*tensor_vector*).
A pointer (*tensor_vector*) to an OpenDR *opendr_tensor_vector_t* along with the pointer into the memory of a vector or array of OpenDR tensors structure (*tensor*) should be provided.
Moreover the number of tensors (*number_of_tensors*) should be included, and it must be better than *1*.

### Function *free_tensor_vector()*
```C
void free_tensor_vector(opendr_tensor_vector_t *tensor_vector);
```
The *free_tensor_vector()* function releases the memory allocated for an OpenDR tensor vector structure (*opendr_tensor_vector_t*).
A pointer (*tensor_vector*) to an OpenDR *opendr_tensor_vector_t* should be provided.

### Function *iter_tensor_vector()*
```C
void iter_tensor_vector(opendr_tensor_t *output, opendr_tensor_vector_t *tensor_vector, int index);
```
The *iter_tensor_vector()* function is used to help the user to iterate the OpenDR tensor vector.
A single OpenDR tensor (*output*) is loaded with the values of the indexed (*index*) tensor of the vector (*tensor_vector*).
A pointer (*tensor_vector*) to an OpenDR *opendr_tensor_vector_t* and an (*index*) along with a pointer (*output*) to an OpenDR *opendr_tensor_t* should be provided.

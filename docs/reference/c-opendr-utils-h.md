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


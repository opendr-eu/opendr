## C_API: data.h


The *data.h* header provides definitions of OpenDR data types that can be used in the C API of OpenDR.

### struct *opendr_image_t*
```C
struct opendr_image {
    void *data;
};
typedef struct opendr_image opendr_image_t;
```


The *opendr_image_t* structure provides a data structure for storing OpenDR images. 
Every function in the C API receiving images is expected to use this structure.
Helper functions that directly convert images into this format are provided in *opendr_utils.h*.

The *opendr_image_t* structure has the following field:

#### `void *data` field

A pointer where image data are stored. 
*opendr_image_t* is using internally OpenCV images (*cv::Mat*) for storing images. 
Therefore, only a pointer to the memory location of the corresponding *cv::Mat* is stored.
Please note that the user is not expected to directly manipulate these data without first converting them into OpenCV data type or using the corresponding functions provided in *opendr_utils.h*.

## C_API: data.h


The *data.h* header provides definitions of OpenDR data types that can be used in the C API of OpenDR.

### struct *OpendrImageT*
```C
struct OpendrImage {
    void *data;
};
typedef struct OpendrImage OpendrImageT;
```


The *OpendrImageT* structure provides a data structure for storing OpenDR images. 
Every function in the C API receiving images is expected to use this structure.
Helper functions that directly convert images into this format are provided in *opendr_utils.h*.

The *OpendrImageT* structure has the following field:

#### `void *data` field

A pointer where image data are stored. 
*OpendrImageT* is using internally OpenCV images (*cv::Mat*) for storing images. 
Therefore, only a pointer to the memory location of the corresponding *cv::Mat* is stored.
Please note that the user is not expected to directly manipulate these data without first converting them into OpenCV data type or using the corresponding functions provided in *opendr_utils.h*.

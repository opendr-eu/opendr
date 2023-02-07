## C_API: data.h


The *data.h* header provides definitions of OpenDR data types that can be used in the C API of OpenDR.

### struct *OpenDRImageT*
```C
struct OpenDRImage {
    void *data;
};
typedef struct OpenDRImage OpenDRImageT;
```


The *OpenDRImageT* structure provides a data structure for storing OpenDR images. 
Every function in the C API receiving images is expected to use this structure.
Helper functions that directly convert images into this format are provided in *opendr_utils.h*.

The *OpenDRImageT* structure has the following field:

#### `void *data` field

A pointer where image data are stored. 
*OpenDRImageT* is internally using OpenCV images (*cv::Mat*) for storing images. 
Therefore, only a pointer to the memory location of the corresponding *cv::Mat* is stored.
Please note that the user is not expected to directly manipulate these data without first converting them into OpenCV data type or using the corresponding functions provided in *OpenDR_utils.h*.

---

### struct *OpenDRTensorT*
```C
struct OpenDRTensor {
  int batchSize;
  int frames;
  int channels;
  int width;
  int height;

  float *data;
};
typedef struct OpenDRTensor OpenDRTensorT;
```


The *OpenDRTensorT* structure provides a data structure for storing OpenDR tensors.
Every function in the C API receiving and returning tensors is expected to use this structure.
Helper functions that directly maps data into this format are provided in *opendr_utils.h*.

The *OpenDRTensorT* structure has the following field:

#### `int batchSize` field

An integer that represents the number of the batch size in the tensor.

#### `int frames` field

An integer that represents the number of frames in the tensor.

#### `int channels` field

An integer that represents the number of channels in the tensor.

#### `int width` field

An integer that represents the width of the tensor.

#### `int height` field

An integer that represents the height of the tensor.

#### `float *data` field

A pointer where data are stored.
*OpenDRTensorT* is internally using a pointer and corresponding sizes to copy the data into the memory of float *data.
Therefore, only a pointer to the memory location of the corresponding data is stored.
Please note that the user is not expected to directly manipulate this data without first converting it into OpenCV cv::Mat or other form of data type or using the corresponding functions provided in *opendr_utils.h*.

### struct *OpenDRTensorVectorT*
```C
struct OpenDRTensorVector {
  int nTensors;
  int *batchSizes;
  int *frames;
  int *channels;
  int *widths;
  int *heights;

  float **memories;
};
typedef struct OpenDRTensorVector OpenDRTensorVectorT;
```


The *OpenDRTensorVectorT* structure provides a data structure for storing OpenDR vector of tensors structures.
Every function in the C API receiving and returning multiple tensors is expected to use this structure.
Helper functions that directly maps data into this format are provided in *opendr_utils.h*.

The *OpenDRTensorVectorT* structure has the following field:

#### `int nTensors` field

An integer that represents the number of tensors in the vector of tensors.

#### `int *batchSizes` field

A pointer of integers that represents the batch size of each tensor.

#### `int *frames` field

A pointer of integers that represents the number of frames in each tensor.

#### `int *channels` field

A pointer of integers that represents the number of channels in each tensor.

#### `int *widths` field

A pointer of integers that represents the width of each tensor.

#### `int *heights` field

A pointer of integers that represents the height of each tensor.

#### `float **datas` field

A pointer where stores the data of each *OpenDRTensorVectorT.data* stored in the vector.
*OpenDRTensorVectorT* is internally using pointers and corresponding sizes to copy the data into the memory of *datas* for each tensor that is provided.
Therefore, only a pointer to the memory location of the corresponding data is stored.
Please note that the user is not expected to directly manipulate this data without first converting it into OpenCV cv::Mat or other form of data type or using the corresponding functions provided in *opendr_utils.h*.

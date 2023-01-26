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

---

### struct *OpendrTensorT*
```C
struct OpendrTensor {
  int batchSize;
  int frames;
  int channels;
  int width;
  int height;

  float *data;
};
typedef struct OpendrTensor OpendrTensorT;
```


The *OpendrTensorT* structure provides a data structure for storing OpenDR tensors.
Every function in the C API receiving and return tensors is expected to use this structure.
Helper functions that directly maps data into this format are provided in *opendr_utils.h*.

The *OpendrTensorT* structure has the following field:

#### `int batchSize` field

An integer that represent the number of batch size in the tensor.

#### `int frames` field

An integer that represent the number of frames in the tensor.

#### `int channels` field

An integer that represent the number of channels in the tensor.

#### `int width` field

An integer that represent the width of the tensor.

#### `int height` field

An integer that represent the height of the tensor.

#### `float *data` field

A pointer where data are stored.
*OpendrTensorT* is using internally a pointer and corresponding sizes to copy the data into the memory of float *data.
Therefore, only a pointer to the memory location of the corresponding data is stored.
Please note that the user is not expected to directly manipulate these data without first converting them into OpenCV cv::Mat or other form of data type or using the corresponding functions provided in *opendr_utils.h*.

### struct *OpendrTensorVectorT*
```C
struct OpendrTensorVector {
  int nTensors;
  int *batchSizes;
  int *frames;
  int *channels;
  int *widths;
  int *heights;

  float **memories;
};
typedef struct OpendrTensorVector OpendrTensorVectorT;
```


The *OpendrTensorVectorT* structure provides a data structure for storing OpenDR vector of tensors structures.
Every function in the C API receiving and returning multiple tensors is expected to use this structure.
Helper functions that directly maps data into this format are provided in *opendr_utils.h*.

The *OpendrTensorVectorT* structure has the following field:

#### `int nTensors` field

An integer that represent the number of tensors in the vector of tensors.

#### `int *batchSizes` field

A pointer of integers that represent the number of batch size in each tensor.

#### `int *frames` field

A pointer of integers that represent the number of frames in each tensor.

#### `int *channels` field

A pointer of integers that represent the number of channels in each tensor.

#### `int *widths` field

A pointer of integers that represent the width of each tensor.

#### `int *heights` field

A pointer of integers that represent the height of each tensor.

#### `float **datas` field

A pointer where stores the data of each *OpendrTensorVectorT.data* stored in the vector.
*OpendrTensorVectorT* is using internally pointers and corresponding sizes to copy the data into the memory of *datas* for each tensor that is provided.
Therefore, only a pointer to the memory location of the corresponding data is stored.
Please note that the user is not expected to directly manipulate these data without first converting them into OpenCV cv::Mat or other form of data type or using the corresponding functions provided in *opendr_utils.h*.

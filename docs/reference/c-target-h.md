## C_API: target.h


The *target.h* header provides definitions of OpenDR targets (inference outputs) that can be used in the C API of OpenDR.

### struct *OpenDRCategoryTargetT*
```C
struct OpenDRCategoryTarget{
    int data;
    float confidence;
};
typedef struct OpenDRCategoryTarget OpenDRCategoryTargetT;
```


The *OpenDRCategoryTargetT* structure provides a data structure for storing inference outputs of classification models.
Every function in the C API that outputs a classification decision is expected to use this structure.

The *OpenDRCategoryTargetT* structure has the following fields:

#### `int data` field

A numerical id of the category to which the input objects belongs to.

#### `float confidence` field

The decision confidence (a value between 0 and 1).


### struct *OpenDRDetectionTargetT*
```C
struct OpenDRDetectionTarget {
  int name;
  float left;
  float top;
  float width;
  float height;
  float score;
};
typedef struct OpenDRDetectionTarget OpenDRDetectionTargetT;
```


The *OpenDRDetectionTargetT* structure provides a data structure for storing inference outputs of detection models.
Every function in the C API that outputs a detection decision is expected to use this structure or a vector of this structure.

The *OpenDRDetectionTargetT* structure has the following fields:

#### `int name` field

A numerical id of the category to which the input objects belongs to.

#### `float left` field

A numerical value that corresponds to the X value of the top-left point of a detection.

#### `float top` field

A numerical value that corresponds to the Y value of the top-left point of a detection.

#### `float width` field

A numerical value that corresponds to the width of a detection.

#### `float height` field

A numerical value that corresponds to the height of a detection.

#### `float score` field

The decision score (a value between 0 and 1).


### struct *OpenDRDetectionVectorTargetT*
```C
struct OpenDRDetectionVectorTarget {
  OpenDRDetectionTargetT *startingPointer;
  int size;
};
typedef struct OpenDRDetectionVectorTarget OpenDRDetectionVectorTargetT;
```


The *OpenDRDetectionVectorTargetT* structure provides a data structure for storing multiple inference outputs of detection models.
Every function in the C API that outputs a detection decision is expected to use this or a *OpenDRDetectionTargetT* structure.

The *OpenDRDetectionVectorTargetT* structure has the following fields:

#### `OpenDRDetectionTargetT startingPointer` field

A pointer to multiple OpenDR detection targets.

#### `int size` field

A numerical value that represents the number of OpenDR detection structures that are stored.

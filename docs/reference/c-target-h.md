## C_API: target.h


The *target.h* header provides definitions of OpenDR targets (inference outputs) that can be used in the C API of OpenDR.

### struct *opendr_category_target_t*
```C
struct opendr_category_target{
    int data;
    float confidence;
};
typedef struct opendr_category_target opendr_category_target_t;
```


The *opendr_category_target_t* structure provides a data structure for storing inference outputs of classification models. 
Every function in the C API that outputs a classification decision is expected to use this structure.

The *opendr_category_target_t* structure has the following field:

#### `int data` field

A numerical id of the category to which the input objects belongs to.

#### `float confidence` field

The decision confidence (a value between 0 and 1).

## C_API: lightweight_open_pose_jit.h


The *lightweight_open_pose.h* header provides function definitions that can be used for accessing the OpenDR pose estimation, lightweight open pose tool.

### Struct *open_pose_model_t*
```C
struct open_pose_model {
 ...
};
typedef struct open_pose_model open_pose_model_t;
```
The *open_pose_model_t* structure keeps all the necessary information that are required by the OpenDR pose estimation lightweight open pose tool (e.g., model weights, normalization information, ONNX session information, etc.).


### Function *load_open_pose_model()*
```C
void load_open_pose_model(char *modelPath, open_pose_model_t *model);
```
 Loads a lightweight open pose, pose estimation model saved in the local filesystem (*modelPath*) in OpenDR format.
 This function also initializes a (*device*) Jit network for performing inference using this model.
 The pre-trained models should follow the OpenDR conventions.
 The Python API can be used to train and export an optimized OpenDR model that can be used for inference using the C API.
 
### Function *free_open_pose_model()*
```C
void free_open_pose_model(open_pose_model_t *model);
```
Releases the memory allocated for a pose estimation lightweight open pose model (*model*).


### Function *forward_open_pose()*
```C
void forward_open_pose(open_pose_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector);
```
This function perform forward pass using a pose estimation, lightweight open pose model (*model*) and an input tensor (*inputTensorValues*).
The function returns an OpenDR tensor vector structure with the forward pass results.


### Function *init_random_opendr_tensor_op()*
```C
void init_random_opendr_tensor_op(opendr_tensor_t *inputTensorValues, open_pose_model_t *model);
```
This is used to initialize a random OpenDR tensor structure (*inputTensorValues*) with the appropriate dimensions for the pose estimation lightweight open pose model (*model*).
The (*model*) keeps all the necessary information.

### Function *init_opendr_tensor_from_img_op()*
```C
void init_opendr_tensor_from_img_op(opendr_image_t *image, opendr_tensor_t *inputTensorValues, open_pose_model_t *model);
```
This is used to initialize an OpenDR tensor structure (*inputTensorValues*) with the data from an OpenDR image (*image*) for the pose estimation lightweight open pose model (*model*).
The (*model*) keeps all the necessary information.

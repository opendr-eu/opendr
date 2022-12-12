/*
 * Copyright 2020-2022 OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef C_API_LIGHTWEIGHT_OPEN_POSE_H
#define C_API_LIGHTWEIGHT_OPEN_POSE_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct open_pose_model {
  // ONNX session objects
  void *onnx_session;
  void *env;
  void *session_options;

  // Sizes for resizing an input image
  int model_size;

  // Statistics for normalization
  float mean_value;
  float img_scale;

  // Model output parameters
  int num_refinement_stages;
  int output_size;
  int stride;

  int even_channel_output;
  int odd_channel_output;
  int batch_size;
  int width_output;
  int height_output;
};
typedef struct open_pose_model open_pose_model_t;

/**
 * Loads a lightweight open pose model saved in OpenDR format
 * @param modelPath path to the OpenDR open_pose model (as exported using OpenDR library)
 * @param model the loaded model
 */
void load_open_pose_model(const char *modelPath, open_pose_model_t *model);

/**
 * This function perform feed forward of open pose model
 * @param model open pose model to be used for inference
 * @param inputTensorValues OpenDR tensor structure as input of the model
 * @param tensorVector OpenDR tensor vector structure to save the output of the feed forward
 */
void forward_open_pose(open_pose_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector);

/**
 * Releases the memory allocated for a open pose model
 * @param model model to be de-allocated
 */
void free_open_pose_model(open_pose_model_t *model);

/**
 * initialize a tensor with values from an opendr image for testing the forward pass of the model
 * @param inputTensorValues opendr tensor to be loaded with random values
 * @param model model to be used for hyper parameters
 */
void init_opendr_tensor_from_img_op(opendr_image_t *image, opendr_tensor_t *inputTensorValues, open_pose_model_t *model);

/**
 * initialize a tensor with random values for testing the forward pass of the model
 * @param inputTensorValues opendr tensor to be loaded with random values
 * @param model model to be used for hyper parameters
 */
void init_random_opendr_tensor_op(opendr_tensor_t *inputTensorValues, open_pose_model_t *model);
#ifdef __cplusplus
}
#endif

#endif  // C_API_LIGHTWEIGHT_OPEN_POSE_H

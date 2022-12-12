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

#ifndef C_API_OBJECT_DETECTION_2D_DETR_H
#define C_API_OBJECT_DETECTION_2D_DETR_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct detr_model {
  // ONNX session objects
  void *onnx_session;
  void *env;
  void *session_options;

  // Sizes for resizing an input image
  int model_size;

  // Statistics for normalization
  float mean_value[3];
  float std_value[3];

  // Recognition threshold
  float threshold;

  // Feature dimension
  int features;
  int output_sizes[2];
};
typedef struct detr_model detr_model_t;

/**
 * Loads a detr object detection model saved in OpenDR format
 * @param modelPath path to the OpenDR detr model (as exported using OpenDR library)
 * @param model the loaded model
 */
void load_detr_model(const char *modelPath, detr_model_t *model);

/**
 * This function perform feed forward of detr object detection model
 * @param model detr object detection model to be used for inference
 * @param inputTensorValues input of the model as OpenCV mat
 * @param tensorVector OpenDR tensor vector structure to save the output of the feed forward
 */
void forward_detr(detr_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector);

/**
 * Releases the memory allocated for a detr object detection model
 * @param model model to be de-allocated
 */
void free_detr_model(detr_model_t *model);

/**
 * initialize a tensor with random values for testing the forward pass of the model
 * @param inputTensorValues opendr tensor to be loaded with random values
 * @param model model to be used for hyper parameters
 */
void init_random_opendr_tensor_detr(opendr_tensor_t *inputTensorValues, detr_model_t *model);

#ifdef __cplusplus
}
#endif

#endif  // C_API_OBJECT_DETECTION_2D_DETR_H

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

#ifndef C_API_FAIR_MOT_OBJECT_TRACKING_2D_H
#define C_API_FAIR_MOT_OBJECT_TRACKING_2D_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct fair_mot_model {
  // ONNX session objects
  void *onnx_session;
  void *env;
  void *session_options;

  // Sizes for resizing an input image
  int model_size[2];
  int in_channels;
  int batch_size;

  // Statistics for normalization
  float mean_value[3];
  float std_value[3];

  // Feature dimension
  int features;
};
typedef struct fair_mot_model fair_mot_model_t;

/**
 * Loads a fair_mot object tracking 2d model saved in OpenDR format
 * @param model_path path to the OpenDR fair_mot model (as exported using OpenDR library)
 * @param model the loaded model
 */
void load_fair_mot_model(const char *model_path, fair_mot_model_t *model);

/**
 * This function perform feed forward of fair_mot object tracking 2d model
 * @param model fair_mot object detection model to be used for inference
 * @param input_tensor_values input of the model as OpenCV mat
 * @param tensor_vector OpenDR tensor vector structure to save the output of the feed forward
 */
void forward_fair_mot(fair_mot_model_t *model, opendr_tensor_t *input_tensor_values, opendr_tensor_vector_t *tensor_vector);

/**
 * Releases the memory allocated for a fair_mot object tracking 2d model
 * @param model model to be de-allocated
 */
void free_fair_mot_model(fair_mot_model_t *model);

#ifdef __cplusplus
}
#endif

#endif  // C_API_FAIR_MOT_OBJECT_TRACKING_2D_H

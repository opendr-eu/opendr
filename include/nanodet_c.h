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

#ifndef C_API_NANODET_H
#define C_API_NANODET_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct nanodet_model {
  // Jit cpp class holder
  void *net;

  // Device to be used
  char *device;

  // Recognition threshold
  float scoreThreshold;

  // Model input size
  int inputSize[2];

  // Keep ratio flag
  int keep_ratio;
};
typedef struct nanodet_model nanodet_model_t;

/**
 * Loads a nanodet object detection model saved in libtorch format
 * @param model_path path to the libtorch nanodet model (as exported using OpenDR library)
 * @param device the device that will be used for the inference
 * @param height the height of model input
 * @param width the width of model input
 * @param scoreThreshold a threshold for score to be infered
 * @param model the loaded model
 */
void load_nanodet_model(char *model_path, char *device, int height, int width, float scoreThreshold, nanodet_model_t *model);

/**
 * This function perform inference using a nanodet object detection model and an input image
 * @param model nanodet model to be used for inference
 * @param image OpenDR image
 * @return vecter of OpenDR bounding box target containing the bounding boxes of the detected objectes
 */
opendr_detection_target_list_t infer_nanodet(opendr_image_t *image, nanodet_model_t *model);

/**
 * Releases the memory allocated for a nanodet object detection model
 * @param model model to be de-allocated
 */
void free_nanodet_model(nanodet_model_t *model);

/**
 * draw the bounding boxes from detections in given image
 * @param opendr_image image that has been used for inference and wanted to be printed
 * @param model nanodet model that has been used for inference
 * @param detections output of the inference
 */
void drawBboxes(opendr_image_t *opendr_image, nanodet_model_t *model, opendr_detection_target_list_t *detections);

#ifdef __cplusplus
}
#endif

#endif  // C_API_NANODET_H

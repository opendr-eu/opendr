/*
 * Copyright 2020-2023 OpenDR European Project
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

struct DetrModel {
  // ONNX session objects
  void *onnxSession;
  void *env;
  void *sessionOptions;

  // Sizes for resizing an input image
  int modelSize;

  // Statistics for normalization
  float meanValue[3];
  float stdValue[3];

  // Recognition threshold
  float threshold;

  // Feature dimension
  int features;
  int outputSizes[2];
};
typedef struct DetrModel DetrModelT;

/**
 * Loads a detr object detection model saved in OpenDR format
 * @param modelPath path to the OpenDR detr model (as exported using OpenDR library)
 * @param model the model to be loaded
 */
void loadDetrModel(const char *modelPath, DetrModelT *model);

/**
 * This function performs feed forward of detr object detection model
 * @param model detr object detection model to be used for feed forward
 * @param tensor OpenDR tensor structure which will be used as input of the model
 * @param vector OpenDR vector of tensors structure to save the output of the feed forward
 */
void forwardDetr(DetrModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);

/**
 * Releases the memory allocated for a detr object detection model
 * @param model model to be de-allocated
 */
void freeDetrModel(DetrModelT *model);

/**
 * This function initializes a tensor with random values for testing the forward pass of the model
 * @param tensor OpenDR tensor structure to be loaded with random values
 * @param model model to be used to initialize the tensor
 */
void initRandomOpendrTensorDetr(OpendrTensorT *tensor, DetrModelT *model);

#ifdef __cplusplus
}
#endif

#endif  // C_API_OBJECT_DETECTION_2D_DETR_H

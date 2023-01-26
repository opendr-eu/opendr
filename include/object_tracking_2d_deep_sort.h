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

#ifndef C_API_DEEP_SORT_OBJECT_TRACKING_2D_H
#define C_API_DEEP_SORT_OBJECT_TRACKING_2D_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct DeepSortModel {
  // ONNX session objects
  void *onnxSession;
  void *env;
  void *sessionOptions;

  // Sizes for resizing an input image
  int modelSize[2];
  int inChannels;
  int batchSize;

  // Statistics for normalization
  float meanValue[3];
  float stdValue[3];

  // Feature dimension
  int features;
};
typedef struct DeepSortModel DeepSortModelT;

/**
 * Loads a deep sort object tracking 2d model saved in OpenDR format
 * @param modelPath path to the OpenDR deep sort model (as exported using OpenDR library)
 * @param model the model to be loaded
 */
void loadDeepSortModel(const char *modelPath, DeepSortModelT *model);

/**
 * This function performs feed forward of deep sort object tracking 2d model
 * @param model deep sort model to be used for feed forward
 * @param tensor OpenDR tensor structure which will be used as input of the model
 * @param vector OpenDR vector of tensors structure to save the output of the feed forward
 */
void forwardDeepSort(DeepSortModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);

/**
 * Releases the memory allocated for a deep sort object tracking 2d model
 * @param model model to be de-allocated
 */
void freeDeepSortModel(DeepSortModelT *model);

/**
 * This function initializes a tensor with random values for testing the forward pass of the model
 * @param tensor OpenDR tensor structure to be loaded with random values
 * @param model model to be used to initialize the tensor
 */
void initRandomOpendrTensorDs(OpendrTensorT *tensor, DeepSortModelT *model);

#ifdef __cplusplus
}
#endif

#endif  // C_API_DEEP_SORT_OBJECT_TRACKING_2D_H

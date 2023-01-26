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

#ifndef C_API_X3D_ACTIVITY_RECOGNITION_H
#define C_API_X3D_ACTIVITY_RECOGNITION_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct X3dModel {
  // ONNX session objects
  void *onnxSession;
  void *env;
  void *sessionOptions;

  // Sizes for resizing an input image
  int modelSize;
  int framesPerClip;
  int inChannels;
  int batchSize;

  // Statistics for normalization
  float meanValue;
  float imgScale;

  // Feature dimension
  int features;
};
typedef struct X3dModel X3dModelT;

/**
 * Loads a x3d activity recognition model saved in OpenDR format
 * @param modelPath path to the OpenDR x3d model (as exported using OpenDR library)
 * @param mode string to determine the model that is used (available options ["xs", "s", "m", "l"])
 * @param model the model to be loaded
 */
void loadX3dModel(const char *modelPath, char *mode, X3dModelT *model);

/**
 * This function performs feed forward of x3d activity recognition model
 * @param model x3d model to be used for feed forward
 * @param tensor OpenDR tensor structure which will be used as input of the model
 * @param vector OpenDR vector of tensors structure to save the output of the feed forward
 */
void forwardX3d(X3dModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);

/**
 * Releases the memory allocated for a x3d activity recognition model
 * @param model model to be de-allocated
 */
void freeX3dModel(X3dModelT *model);

/**
 * This function initializes a tensor with random values for testing the forward pass of the model
 * @param tensor OpenDR tensor structure to be loaded with random values
 * @param model model to be used to initialize the tensor
 */
void initRandomOpendrTensorX3d(OpendrTensorT *tensor, X3dModelT *model);

#ifdef __cplusplus
}
#endif

#endif  // C_API_X3D_ACTIVITY_RECOGNITION_H

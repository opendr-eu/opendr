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

#ifndef C_API_LIGHTWEIGHT_OPEN_POSE_H
#define C_API_LIGHTWEIGHT_OPEN_POSE_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct OpenPoseModel {
  // ONNX session objects
  void *onnxSession;
  void *env;
  void *sessionOptions;

  // Sizes for resizing an input image
  int modelSize;

  // Statistics for normalization
  float meanValue;
  float imgScale;

  // Model output parameters
  int nRefinementStages;
  int outputSize;
  int stride;

  int evenChannelOutput;
  int oddChannelOutput;
  int batchSize;
  int widthOutput;
  int heightOutput;
};
typedef struct OpenPoseModel OpenPoseModelT;

/**
 * Loads a lightweight open pose model saved in OpenDR format
 * @param modelPath path to the OpenDR open pose model (as exported using OpenDR library)
 * @param model the model to be loaded
 */
void loadOpenPoseModel(const char *modelPath, OpenPoseModelT *model);

/**
 * This function performs feed forward of open pose model
 * @param model open pose model to be used for feed forward
 * @param tensor OpenDR tensor structure which will be used as input of the model
 * @param vector OpenDR vector of tensors structure to save the output of the feed forward
 */
void forwardOpenPose(OpenPoseModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector);

/**
 * Releases the memory allocated for an open pose model
 * @param model model to be de-allocated
 */
void freeOpenPoseModel(OpenPoseModelT *model);

/**
 * This function initializes a tensor with values from an OpenDR image for testing the forward pass of the model
 * @param image OpenDR image to load into tensor
 * @param tensor OpenDR tensor structure to be loaded with the image values
 * @param model model to be used to initialize the tensor
 */
void initOpendrTensorFromImgOp(OpendrImageT *image, OpendrTensorT *tensor, OpenPoseModelT *model);

/**
 * This function initializes a tensor with random values for testing the forward pass of the model
 * @param tensor OpenDR tensor structure to be loaded with random values
 * @param model model to be used to initialize the tensor
 */
void initRandomOpendrTensorOp(OpendrTensorT *tensor, OpenPoseModelT *model);
#ifdef __cplusplus
}
#endif

#endif  // C_API_LIGHTWEIGHT_OPEN_POSE_H

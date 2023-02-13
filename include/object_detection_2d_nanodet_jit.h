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

#ifndef C_API_NANODET_H
#define C_API_NANODET_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct NanodetModel {
  // Jit cpp class holder
  void *network;

  // Device to be used
  char *device;
  int **colorList;
  int numberOfClasses;

  // Recognition threshold
  float scoreThreshold;

  // Model input size
  int inputSizes[2];

  // Keep ratio flag
  int keepRatio;
};
typedef struct NanodetModel NanodetModelT;

/**
 * Loads a nanodet object detection model saved in libtorch format.
 * @param modelPath path to the libtorch nanodet model (as exported using OpenDR)
 * @param modelName name of the model to be loaded
 * @param device the device that will be used for inference
 * @param scoreThreshold confidence threshold
 * @param height the height of model input, if set to zero the trained height will be used instead
 * @param width the width of model input, if set to zero the trained width will be used instead
 * @param model the model to be loaded
 */
void loadNanodetModel(const char *modelPath, const char *modelName, const char *device, float scoreThreshold, int height,
                      int width, NanodetModelT *model);

/**
 * This function performs inference using a nanodet object detection model and an input image.
 * @param model nanodet model to be used for inference
 * @param image OpenDR image
 * @return OpenDR detection vector target containing the detections of the recognized objects
 */
OpenDRDetectionVectorTargetT inferNanodet(NanodetModelT *model, OpenDRImageT *image);

/**
 * Releases the memory allocated for a nanodet object detection model.
 * @param model model to be de-allocated
 */
void freeNanodetModel(NanodetModelT *model);

/**
 * Draw the bounding boxes from detections in the given image.
 * @param image image that has been used for inference
 * @param model nanodet model that has been used for inference
 * @param detectionsVector output of the inference
 * @param show if the value given is zero, the image will be displayed until a key is pressed
 */
void drawBboxes(OpenDRImageT *image, NanodetModelT *model, OpenDRDetectionVectorTargetT *vector, int show);

#ifdef __cplusplus
}
#endif

#endif  // C_API_NANODET_H

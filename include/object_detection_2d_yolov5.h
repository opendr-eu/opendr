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

#ifndef C_API_YOLOV5_H
#define C_API_YOLOV5_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct Yolov5Model {
  // ONNX session objects
  void *onnxSession;
  void *env;
  void *sessionOptions;

  // Inference parameters
  float confThreshold;
  float iouThreshold;
  OpenDRStringsVectorT labels;
  int **colorList;
  int numberOfClasses;

  // Model parameters
  int inputSizes[2];
  int isDynamicInputShape;

};
typedef struct Yolov5Model Yolov5ModelT;

/**
 * Loads a yolov5 object detection model saved in libtorch format.
 * @param modelPath path to the libtorch yolov5 model (as exported using OpenDR)
 * @param modelName name of the model to be loaded
 * @param device the device that will be used for inference
 * @param scoreThreshold confidence threshold
 * @param height the height of model input, if set to zero the trained height will be used instead
 * @param width the width of model input, if set to zero the trained width will be used instead
 * @param model the model to be loaded
 */
void loadYolov5Model(const char *modelPath, const char *modelName, const char *device, float confThreshold, float iouThreshold,
                     int height, int width, Yolov5ModelT *model);

/**
 * This function performs inference using a yolov5 object detection model and an input image.
 * @param model yolov5 model to be used for inference
 * @param image OpenDR image
 * @return OpenDR detection vector target containing the detections of the recognized objects
 */
OpenDRDetectionVectorTargetT inferYolov5(Yolov5ModelT *model, OpenDRImageT *image);

/**
 * Releases the memory allocated for a yolov5 object detection model.
 * @param model model to be de-allocated
 */
void freeYolov5Model(Yolov5ModelT *model);

///**
// * Draw the bounding boxes from detections in the given image.
// * @param image image that has been used for inference
// * @param model yolov5 model that has been used for inference
// * @param vector output of the inference
// * @param show if the value given is zero, the image will be displayed until a key is pressed
// */
//void drawBboxesYolov5(OpenDRImageT *image, Yolov5ModelT *model, OpenDRDetectionVectorTargetT *vector, int show);

#ifdef __cplusplus
}
#endif

#endif  // C_API_YOLOV5_H

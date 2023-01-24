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

#ifndef C_API_FACE_RECOGNITION_H
#define C_API_FACE_RECOGNITION_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct FaceRecognitionModel {
  // ONNX session objects
  void *onnxSession;
  void *env;
  void *sessionOptions;

  // Sizes for resizing and cropping an input image
  int modelSize;
  int resizeSize;

  // Statistics for normalization
  float meanValue;
  float stdValue;

  // Recognition threshold
  float threshold;

  // Feature dimension
  int outputSize;

  // Database data
  void *database;
  int *databaseIds;
  char **personNames;

  // Number of persons in the database
  int nPersons;
  // Number of features vectors in the database
  int nFeatures;
};
typedef struct FaceRecognitionModel FaceRecognitionModelT;

/**
 * Loads a face recognition model saved in OpenDR format.
 * @param modelPath path to the OpenDR face recognition model (as exported using OpenDR library)
 * @param model the loaded model
 */
void loadFaceRecognitionModel(const char *modelPath, FaceRecognitionModelT *model);

/**
 * This function perform inference using a face recognition model and an input image.
 * @param model face recognition model to be used for inference
 * @param image OpenDR image
 * @return OpenDR classification target containing the id of the recognized person
 */
OpendrCategoryTargetT inferFaceRecognition(FaceRecognitionModelT *model, OpendrImageT *image);

/**
 * Builds a face recognition database (containing images for persons to be recognized). This function expects the
 * databaseFolder to have the same format as the main Python toolkit.
 * @param databaseFolder folder containing the database
 * @param outputPath output path to store the binary database. This file should be loaded along with the face
 * recognition model before performing inference.
 * @param model the face recognition model to be used for extracting the database features
 */
void buildDatabaseFaceRecognition(const char *databaseFolder, const char *outputPath, FaceRecognitionModelT *model);

/**
 * Loads an already built database into the face recognition model. After this step, the model can be used for
 * performing inference.
 * @param databasePath path to the database file
 * @param model the face recognition model to be used for inference
 */
void loadDatabaseFaceRecognition(const char *databasePath, FaceRecognitionModelT *model);

/**
 * Returns the name of a recognition person by decoding the category id into a string.
 * @param model the face recognition model to be used for inference
 * @param category the predicted category
 * @param personName buffer to store the person name
 */
void decodeCategoryFaceRecognition(FaceRecognitionModelT *model, OpendrCategoryTargetT category, char *personName);

/**
 * Releases the memory allocated for a face recognition model.
 * @param model model to be de-allocated
 */
void freeFaceRecognitionModel(FaceRecognitionModelT *model);

#ifdef __cplusplus
}
#endif

#endif  // C_API_FACE_RECOGNITION_H

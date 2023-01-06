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

#include "face_recognition.h"
#include "opendr_utils.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  // Create a face recognition model
  FaceRecognitionModelT model;
  // Load a pretrained model (see instructions for downloading the data)
  loadFaceRecognitionModel("data/optimized_model", &model);

  // Build and load the database
  buildDatabaseFaceRecognition("data/database", "data/database.dat", &model);
  loadDatabaseFaceRecognition("data/database.dat", &model);

  // Load an image and performance inference
  OpendrImageT image;
  loadImage("data/database/1/1.jpg", &image);
  if (!image.data) {
    printf("Image not found!");
    return 1;
  }
  OpendrCategoryTargetT res = inferFaceRecognition(&model, &image);
  // Free the image resources
  freeImage(&image);

  // Get the prediction and decode it
  char buff[512];
  decodeCategoryFaceRecognition(&model, res, buff);
  printf("Predicted category %d (folder name: %s) with confidence %f\n", res.data, buff, res.confidence);

  // Free the model resources
  freeFaceRecognitionModel(&model);

  return 0;
}

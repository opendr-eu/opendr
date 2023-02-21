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


#include <stdio.h>
#include <stdlib.h>
#include "object_detection_2d_nanodet_jit.h"
#include "opendr_utils.h"

int main(int argc, char **argv) {
  NanodetModelT model;

  printf("start init model\n");
  loadNanodetModel("./data/object_detection_2d/nanodet/optimized_model", "m", "cuda", 0.35, 0, 0, &model);
  printf("success\n");

  OpenDRImageT image;

  loadImage("data/object_detection_2d/nanodet/database/000000000036.jpg", &image);
  if (!image.data) {
    printf("Image not found!");
    return 1;
  }

  // Initialize OpenDR detection target list;
  OpenDRDetectionVectorTargetT results;
  initDetectionsVector(&results);

  results = inferNanodet(&model, &image);

  drawBboxes(&image, &model, &results, 0);

  // Free the memory
  freeDetectionsVector(&results);
  freeImage(&image);
  freeNanodetModel(&model);

  return 0;
}

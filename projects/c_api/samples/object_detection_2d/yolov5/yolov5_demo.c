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
#include "object_detection_2d_yolov5.h"
#include "opendr_utils.h"

int main(int argc, char **argv) {
  Yolov5ModelT model;

  printf("start init model\n");
  loadYolov5Model("./data/object_detection_2d/yolov5/optimized_model", "s", "cpu", 0, 0, 0, 0, &model);
  printf("success\n");

  OpenDRImageT image;

  loadImage("data/object_detection_2d/yolov5/database/zidane.jpg", &image);
  if (!image.data) {
    printf("Image not found!");
    return 1;
  }

  // Initialize OpenDR detection target list;
  OpenDRDetectionVectorTargetT results;
  initDetectionsVector(&results);

  results = inferYolov5(&model, &image);

  drawBboxes(&image, &results, &model.labels, model.colorList, 0);

  // Free the memory
  freeDetectionsVector(&results);
  freeYolov5Model(&model);
  freeImage(&image);

  return 0;
}

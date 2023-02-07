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

#include <opencv2/opencv.hpp>
#include <cstdlib>

int main(int argc, char **argv) {
  NanodetModelT model;

  printf("start init model\n");
  loadNanodetModel("./data/object_detection_2d/nanodet/new_opt_model", "m", "cuda", 0.35, 0, 0, &model);
  printf("success\n");

  int height = 320;
  int width = 320;
  std::srand(1);

  OpenDRImageT opImage;

  loadImage("data/object_detection_2d/nanodet/database/000000000036.jpg", &opImage);

//  cv::Mat frame(height,width,CV_8UC3);
//  for(int i = 0; i < frame.rows; i++) {
//    for(int j = 0; j < frame.cols; j++) {
//      frame.at<cv::Vec3b>(i, j)[0] = rand() % 256;
//      frame.at<cv::Vec3b>(i, j)[1] = rand() % 256;
//      frame.at<cv::Vec3b>(i, j)[2] = rand() % 256;
//    }
//  }
//
//  OpenDRImageT opImage;
//  // Add frame data to OpenDR Image
//  if (frame.empty()) {
//    opImage.data = NULL;
//  } else {
//    cv::Mat *tempMatPtr = new cv::Mat(frame);
//    opImage.data = (void *)tempMatPtr;
//  }

  int repetitions = 1000;
  int warmup = 100;
  benchmarkNanodet(&model, &opImage, repetitions, warmup);
  // Free the memory
  freeNanodetModel(&model);

  return 0;
}

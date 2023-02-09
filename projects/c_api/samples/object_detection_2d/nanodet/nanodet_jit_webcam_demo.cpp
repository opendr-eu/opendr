// Copyright 2020-2023 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "object_detection_2d_nanodet_jit.h"
#include "opendr_utils.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char **argv) {
  NanodetModelT model;

  printf("start init model\n");
  loadNanodetModel("./data/object_detection_2d/nanodet/optimized_model", "m", "cuda", 0.35, 0, 0, &model);
  printf("success\n");

  cv::Mat frameCap;
  cv::Mat frame;
  OpenDRImageT opImage;
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
    return -1;
  }

  // Initialize OpenDR detection target list;
  OpenDRDetectionVectorTargetT results;
  initDetectionsVector(&results);

  double avg_fps = 0.0;
  while (true) {
    cap >> frameCap;

    cv::resize(frameCap, frame, cv::Size(640, 640), 0, 0, cv::INTER_CUBIC);

    // Add frame data to OpenDR Image
    if (frame.empty()) {
      opImage.data = NULL;
    } else {
      cv::Mat *tempMatPtr = new cv::Mat(frame);
      opImage.data = (void *)tempMatPtr;
    }


    auto start = std::chrono::steady_clock::now();
    results = inferNanodet(&model, &opImage);
    auto end = std::chrono::steady_clock::now();
    double fps = 1000000000.0 / ((double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));

    avg_fps = fps * 0.8 + avg_fps * 0.2;

    drawBboxes(&opImage, &model, &results, 1);
    cv::Mat *opencvImage = static_cast<cv::Mat *>(opImage.data);

    // Put fps counter
    std::string fpsText = "FPS: " + std::to_string(fps);
    cv::putText(*opencvImage, fpsText, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    cv::imshow("live view", *opencvImage);
    if (cv::waitKey(1) >= 0)
      break;
  }

  // Free the memory
  freeDetectionsVector(&results);
  freeNanodetModel(&model);

  return 0;
}

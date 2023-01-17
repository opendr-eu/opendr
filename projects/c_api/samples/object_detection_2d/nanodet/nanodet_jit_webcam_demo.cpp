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
#include <chrono>
#include "object_detection_2d_nanodet_jit.h"
#include "opendr_utils.h"

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr,
            "usage: %s [model_path] [device] [images_path] [input_sizes].\n"
            "model_path = path/to/your/libtorch/model.pth \ndevice = cuda or cpu \n"
            "images_path = \"xxx/xxx.jpg\" \ninput_size = width height.\n",
            argv[0]);
    return -1;
  }

  NanodetModelT model;

  int height = atoi(argv[4]);
  int width = atoi(argv[5]);
  printf("start init model\n");
  loadNanodetModel(argv[1], argv[2], height, width, 0.35, &model);
  printf("success\n");

  cv::Mat frameCap;
  cv::Mat frame;
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "ERROR! Unable to open camera\n";
    return -1;
  }

  // Initialize opendr detection target list;
  OpendrDetectionVectorTargetT results;
  initDetectionsVector(&results);
  double fps;
  double avg_fps = 0.0;
  int count = 0;

  clock_t start_time, end_time;
  while (count > -1) {
    cap >> frameCap;

    cv::resize(frameCap, frame, cv::Size(640, 640), 0, 0, cv::INTER_CUBIC);
    auto start = std::chrono::steady_clock::now();
    results = inferNanodet(&model, &frame);
    auto end = std::chrono::steady_clock::now();
    fps = 1000000000.0 / ((double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));

    avg_fps = fps * 0.8 + avg_fps * 0.2;
    if (count > 5.0) {
      drawBboxesWithFps(&frame, &model, &results, avg_fps);
    }
    count += 1;

    if (cv::waitKey(1) >= 0)
      break;
  }

  // Free the memory
  freeDetectionsVector(&results);
  freeNanodetModel(&model);

  return 0;
}

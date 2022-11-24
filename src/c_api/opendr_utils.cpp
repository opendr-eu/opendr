//
// Copyright 2020-2022 OpenDR European Project
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

#include "opendr_utils.h"
#include "data.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void load_image(const char *path, opendr_image_t *image) {
  cv::Mat opencv_image = cv::imread(path, cv::IMREAD_COLOR);
  if (opencv_image.empty()) {
    image->data = NULL;
  } else {
    image->data = new cv::Mat(opencv_image);
  }
}

void free_image(opendr_image_t *image) {
  if (image->data) {
    cv::Mat *opencv_image = static_cast<cv::Mat *>(image->data);
    delete opencv_image;
  }
}

void initialize_detections(opendr_detection_target_list_t *detections) {
  std::vector<opendr_detection_target> dets;
  opendr_detection_target_t det;
  det.name = -1;
  det.left = 0.0;
  det.top = 0.0;
  det.width = 0.0;
  det.height = 0.0;
  det.score = 0.0;
  dets.push_back(det);

  load_detections(&detections, dets.data(), (int)dets.size());
}

void load_detections(opendr_detection_target_list_t *detections, opendr_detection_target_t *vectorDataPtr, int vectorSize) {
  detections->size = vectorSize;
  int sizeOfOutput = (vectorSize) * sizeof(opendr_detection_target_t);
  detections->starting_pointer = (opendr_detection_target_t *)malloc(sizeOfOutput);
  std::memcpy(detections->starting_pointer, vectorDataPtr, sizeOfOutput);
}

void free_detections(opendr_detection_target_list_t *detections) {
  if (detections->starting_pointer != NULL)
    free(detections->starting_pointer);
}
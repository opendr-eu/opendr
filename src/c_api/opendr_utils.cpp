//
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

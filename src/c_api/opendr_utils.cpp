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

#include <document.h>
#include <stringbuffer.h>
#include <writer.h>

const char *json_get_key_string(const char *json, const char *key) {
  rapidjson::Document doc;
  doc.Parse(json);
  if ((!doc.IsObject()) || (!doc.HasMember(key))) {
    return "";
  }
  const rapidjson::Value &value = doc[key];
  if (!value.IsString()) {
    return "";
  }
  return value.GetString();
}

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

void init_detections_vector(opendr_detection_vector_target_t *detection_vector) {
  detection_vector->starting_pointer = NULL;

  std::vector<opendr_detection_target> detections;
  opendr_detection_target_t detection;

  detection.name = -1;
  detection.left = 0.0;
  detection.top = 0.0;
  detection.width = 0.0;
  detection.height = 0.0;
  detection.score = 0.0;

  detections.push_back(detection);

  load_detections_vector(detection_vector, detections.data(), static_cast<int>(detections.size()));
}

void load_detections_vector(opendr_detection_vector_target_t *detection_vector, opendr_detection_target_t *detection,
                            int vector_size) {
  free_detections_vector(detection_vector);

  detection_vector->size = vector_size;
  int size_of_output = (vector_size) * sizeof(opendr_detection_target_t);
  detection_vector->starting_pointer = static_cast<opendr_detection_target_t *>(malloc(size_of_output));
  std::memcpy(detection_vector->starting_pointer, detection, size_of_output);
}

void free_detections_vector(opendr_detection_vector_target_t *detection_vector) {
  if (detection_vector->starting_pointer != NULL)
    free(detection_vector->starting_pointer);
}

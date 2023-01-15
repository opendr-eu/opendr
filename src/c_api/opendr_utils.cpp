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
#include <opencv2/videoio.hpp>
#include <document.h>
#include <stringbuffer.h>
#include <writer.h>
#include <iostream>

float jsonGetKeyFromInferenceParams(const char *json, const char *key, const int index) {
  rapidjson::Document doc;
  doc.Parse(json);
  if ((!doc.IsObject()) || (!doc.HasMember("inference_params"))) {
    return 0.0f;
  }
  const rapidjson::Value &inferenceParams = doc["inference_params"];
  if ((!inferenceParams.IsObject()) || (!inferenceParams.HasMember(key))) {
    return 0.0f;
  }
  const rapidjson::Value &value = inferenceParams[key];
  if (value.IsArray()) {
    if (value.Size() <= index) {
      return 0.0f;
    }
    if (!value[index].IsFloat()) {
      return 0.0f;
    }
    return value[index].GetFloat();
  }
  if (!value.IsFloat()) {
    return 0.0f;
  }
  return value.GetFloat();
}

const char *jsonGetKeyString(const char *json, const char *key, const int index) {
  rapidjson::Document doc;
  doc.Parse(json);
  if ((!doc.IsObject()) || (!doc.HasMember(key))) {
    return "";
  }
  const rapidjson::Value &value = doc[key];
  if (value.IsArray()) {
    if (value.Size() <= index) {
      return "";
    }
    if (!value[index].IsString()) {
      return "";
    }
    return value[index].GetString();
  }
  if (!value.IsString()) {
    return "";
  }
  return value.GetString();
}

float jsonGetKeyFloat(const char *json, const char *key, const int index) {
  rapidjson::Document doc;
  doc.Parse(json);
  if ((!doc.IsObject()) || (!doc.HasMember(key))) {
    return 0.0f;
  }
  const rapidjson::Value &value = doc[key];
  if (value.IsArray()) {
    if (value.Size() <= index) {
      return 0.0f;
    }
    if (!value[index].IsFloat()) {
      return 0.0f;
    }
    return value[index].IsFloat();
  }
  if (!value.IsFloat()) {
    return 0.0f;
  }
  return value.GetFloat();
}

void loadImage(const char *path, OpendrImageT *image) {
  cv::Mat opencvImage = cv::imread(path, cv::IMREAD_COLOR);
  if (opencvImage.empty()) {
    image->data = NULL;
  } else {
    image->data = new cv::Mat(opencvImage);
  }
}

void creatCamera(int cameraId, int width, int height, OpendrCameraT *camera) {
  camera = (OpendrCameraT*)malloc(sizeof(OpendrCameraT));
  camera->cap = new cv::VideoCapture(cameraId);
  camera->cameraId = cameraId;
  camera->width = width;
  camera->height = height;
}

void freeCamera(OpendrCameraT *camera) {
  cv::VideoCapture* capture = (cv::VideoCapture*)camera->cap;
  capture->release();
  free(camera);
}

void loadImageFromCapture(OpendrCameraT *camera, OpendrImageT *image) {
  cv::VideoCapture* capture = (cv::VideoCapture*)camera->cap;

  if (!capture->isOpened()) {
    std::cerr << "Error: Unable to open the camera" << std::endl;
    return;
  }
  capture->set(cv::CAP_PROP_FRAME_WIDTH,camera->width);
  capture->set(cv::CAP_PROP_FRAME_HEIGHT,camera->height);

  cv::Mat opencvImage;
  *capture >> opencvImage;
  if (opencvImage.empty()) {
    image->data = NULL;
  } else {
    image->data = new cv::Mat(opencvImage);
  }
}

void freeImage(OpendrImageT *image) {
  if (image->data) {
    cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
    delete opencvImage;
  }
}

void initDetectionsVector(OpendrDetectionVectorTargetT *detectionVector) {
  detectionVector->startingPointer = NULL;

  std::vector<OpendrDetectionTarget> detections;
  OpendrDetectionTargetT detection;

  detection.name = -1;
  detection.left = 0.0;
  detection.top = 0.0;
  detection.width = 0.0;
  detection.height = 0.0;
  detection.score = 0.0;

  detections.push_back(detection);

  loadDetectionsVector(detectionVector, detections.data(), static_cast<int>(detections.size()));
}

void loadDetectionsVector(OpendrDetectionVectorTargetT *detectionVector, OpendrDetectionTargetT *detection, int vectorSize) {
  freeDetectionsVector(detectionVector);

  detectionVector->size = vectorSize;
  int sizeOfOutput = (vectorSize) * sizeof(OpendrDetectionTargetT);
  detectionVector->startingPointer = static_cast<OpendrDetectionTargetT *>(malloc(sizeOfOutput));
  std::memcpy(detectionVector->startingPointer, detection, sizeOfOutput);
}

void freeDetectionsVector(OpendrDetectionVectorTargetT *detectionVector) {
  if (detectionVector->startingPointer != NULL)
    free(detectionVector->startingPointer);
}

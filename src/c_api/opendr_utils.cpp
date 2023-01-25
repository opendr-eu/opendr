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

#include <document.h>
#include <stringbuffer.h>
#include <writer.h>

const char *jsonGetStringFromKey(const char *json, const char *key, const int index) {
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

float jsonGetFloatFromKey(const char *json, const char *key, const int index) {
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

const char *jsonGetStringFromKeyInInferenceParams(const char *json, const char *key, const int index) {
  rapidjson::Document doc;
  doc.Parse(json);
  if ((!doc.IsObject()) || (!doc.HasMember("inference_params"))) {
    return "";
  }
  const rapidjson::Value &inferenceParams = doc["inference_params"];
  if ((!inferenceParams.IsObject()) || (!inferenceParams.HasMember(key))) {
    return "";
  }
  const rapidjson::Value &value = inferenceParams[key];
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

float jsonGetFloatFromKeyInInferenceParams(const char *json, const char *key, const int index) {
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

void loadImage(const char *path, OpendrImageT *image) {
  cv::Mat opencvImage = cv::imread(path, cv::IMREAD_COLOR);
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

void initTensor(OpendrTensorT *tensor) {
  tensor->batchSize = 0;
  tensor->frames = 0;
  tensor->channels = 0;
  tensor->width = 0;
  tensor->height = 0;
  tensor->data = NULL;
}

void loadTensor(OpendrTensorT *tensor, void *tensorData, int batchSize, int frames, int channels, int width, int height) {
  freeTensor(tensor);

  tensor->batchSize = batchSize;
  tensor->frames = frames;
  tensor->channels = channels;
  tensor->width = width;
  tensor->height = height;

  int sizeOfData = (batchSize * frames * channels * width * height) * sizeof(float);
  tensor->data = static_cast<float *>(malloc(sizeOfData));
  std::memcpy(tensor->data, tensorData, sizeOfData);
}

void freeTensor(OpendrTensorT *tensor) {
  if (tensor->data != NULL) {
    free(tensor->data);
    tensor->data = NULL;
  }
}

void initTensorVector(OpendrTensorVectorT *vector) {
  vector->nTensors = 0;
  vector->batchSizes = NULL;
  vector->frames = NULL;
  vector->channels = NULL;
  vector->widths = NULL;
  vector->heights = NULL;
  vector->memories = NULL;
}

void loadTensorVector(OpendrTensorVectorT *vector, OpendrTensorT *tensor, int nTensors) {
  freeTensorVector(vector);

  vector->nTensors = nTensors;
  int sizeOfDataShape = nTensors * sizeof(int);
  /* initialize arrays to hold size values for each tensor */
  vector->batchSizes = static_cast<int *>(malloc(sizeOfDataShape));
  vector->frames = static_cast<int *>(malloc(sizeOfDataShape));
  vector->channels = static_cast<int *>(malloc(sizeOfDataShape));
  vector->widths = static_cast<int *>(malloc(sizeOfDataShape));
  vector->heights = static_cast<int *>(malloc(sizeOfDataShape));

  /* initialize array to hold data values for all tensors */
  vector->memories = static_cast<float **>(malloc(nTensors * sizeof(float *)));

  /* copy size values */
  for (int i = 0; i < number_of_tensors; i++) {
    (vector->batchSizes)[i] = tensor[i].batchSize;
    (vector->frames)[i] = tensor[i].frames;
    (vector->channels)[i] = tensor[i].channels;
    (vector->widths)[i] = tensor[i].width;
    (vector->heights)[i] = tensor[i].height;

    /* copy data values by,
     * initialize a data pointer into a tensor,
     * copy the values,
     * set tensor data pointer to watch the memory pointer*/
    int sizeOfData = ((tensor[i].batchSizes) * (tensor[i].frames) * (tensor[i].channels) * (tensor[i].width) *
                      (tensor[i].height) * sizeof(float));
    float *memoryOfDataTensor = static_cast<float *>(malloc(sizeOfData));
    std::memcpy(memoryOfDataTensor, tensor[i].data, sizeOfData);
    (vector->memories)[i] = memoryOfDataTensor;
  }
}

void freeTensorVector(OpendrTensorVectorT *vector) {
  // free vector pointers
  if (vector->batchSizes != NULL) {
    free(vector->batchSizes);
    vector->batchSizes = NULL;
  }
  if (vector->frames != NULL) {
    free(vector->frames);
    vector->frames = NULL;
  }
  if (vector->channels != NULL) {
    free(vector->channels);
    vector->channels = NULL;
  }
  if (vector->widths != NULL) {
    free(vector->widths);
    vector->widths = NULL;
  }
  if (vector->heights != NULL) {
    free(vector->heights);
    vector->heights = NULL;
  }

  // free tensors data and vector memory
  if (vector->memories != NULL) {
    free(vector->memories);
    vector->memories = NULL;
  }

  // reset tensor vector values
  vector->nTensors = 0;
}

void iterTensorVector(OpendrTensorT *tensor, OpendrTensorVectorT *vector, int index) {
  loadTensor(tensor, static_cast<void *>((vector->memories)[index]), (vector->batchSizes)[index], (vector->frames)[index],
             (vector->channels)[index], (vector->widths)[index], (vector->heights)[index]);
}

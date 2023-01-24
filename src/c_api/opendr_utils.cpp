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

void init_tensor(opendr_tensor_t *opendr_tensor) {
  opendr_tensor->batch_size = 0;
  opendr_tensor->frames = 0;
  opendr_tensor->channels = 0;
  opendr_tensor->width = 0;
  opendr_tensor->height = 0;
  opendr_tensor->data = NULL;
}

void load_tensor(opendr_tensor_t *opendr_tensor, void *tensor_data, int batch_size, int frames, int channels, int width,
                 int height) {
  free_tensor(opendr_tensor);

  opendr_tensor->batch_size = batch_size;
  opendr_tensor->frames = frames;
  opendr_tensor->channels = channels;
  opendr_tensor->width = width;
  opendr_tensor->height = height;

  int size_of_data = (batch_size * frames * channels * width * height) * sizeof(float);
  opendr_tensor->data = static_cast<float *>(malloc(size_of_data));
  std::memcpy(opendr_tensor->data, tensor_data, size_of_data);
}

void free_tensor(opendr_tensor_t *opendr_tensor) {
  if (opendr_tensor->data != NULL) {
    free(opendr_tensor->data);
    opendr_tensor->data = NULL;
  }
}

void init_tensor_vector(opendr_tensor_vector_t *tensor_vector) {
  tensor_vector->n_tensors = 0;
  tensor_vector->batch_sizes = NULL;
  tensor_vector->frames = NULL;
  tensor_vector->channels = NULL;
  tensor_vector->widths = NULL;
  tensor_vector->heights = NULL;
  tensor_vector->memories = NULL;
}

void load_tensor_vector(opendr_tensor_vector_t *tensor_vector, opendr_tensor_t *tensor, int number_of_tensors) {
  free_tensor_vector(tensor_vector);

  tensor_vector->n_tensors = number_of_tensors;
  int size_of_shape_data = number_of_tensors * sizeof(int);
  /* initialize arrays to hold size values for each tensor */
  tensor_vector->batch_sizes = static_cast<int *>(malloc(size_of_shape_data));
  tensor_vector->frames = static_cast<int *>(malloc(size_of_shape_data));
  tensor_vector->channels = static_cast<int *>(malloc(size_of_shape_data));
  tensor_vector->widths = static_cast<int *>(malloc(size_of_shape_data));
  tensor_vector->heights = static_cast<int *>(malloc(size_of_shape_data));

  /* initialize array to hold data values for all tensors */
  tensor_vector->memories = static_cast<float **>(malloc(number_of_tensors * sizeof(float *)));

  /* copy size values */
  for (int i = 0; i < number_of_tensors; i++) {
    (tensor_vector->batch_sizes)[i] = tensor[i].batch_size;
    (tensor_vector->frames)[i] = tensor[i].frames;
    (tensor_vector->channels)[i] = tensor[i].channels;
    (tensor_vector->widths)[i] = tensor[i].width;
    (tensor_vector->heights)[i] = tensor[i].height;

    /* copy data values by,
     * initialize a data pointer into a tensor,
     * copy the values,
     * set tensor data pointer to watch the memory pointer*/
    int size_of_data = ((tensor[i].batch_size) * (tensor[i].frames) * (tensor[i].channels) * (tensor[i].width) *
                        (tensor[i].height) * sizeof(float));
    float *memory_of_data_tensor = static_cast<float *>(malloc(size_of_data));
    std::memcpy(memory_of_data_tensor, tensor[i].data, size_of_data);
    (tensor_vector->memories)[i] = memory_of_data_tensor;
  }
}

void free_tensor_vector(opendr_tensor_vector_t *tensor_vector) {
  // free vector pointers
  if (tensor_vector->batch_sizes != NULL) {
    free(tensor_vector->batch_sizes);
    tensor_vector->batch_sizes = NULL;
  }
  if (tensor_vector->frames != NULL) {
    free(tensor_vector->frames);
    tensor_vector->frames = NULL;
  }
  if (tensor_vector->channels != NULL) {
    free(tensor_vector->channels);
    tensor_vector->channels = NULL;
  }
  if (tensor_vector->widths != NULL) {
    free(tensor_vector->widths);
    tensor_vector->widths = NULL;
  }
  if (tensor_vector->heights != NULL) {
    free(tensor_vector->heights);
    tensor_vector->heights = NULL;
  }

  // free tensors data and vector memory
  if (tensor_vector->memories != NULL) {
    free(tensor_vector->memories);
    tensor_vector->memories = NULL;
  }

  // reset tensor vector values
  tensor_vector->n_tensors = 0;
}

void iter_tensor_vector(opendr_tensor_t *output, opendr_tensor_vector_t *tensor_vector, int index) {
  load_tensor(output, static_cast<void *>((tensor_vector->memories)[index]), (tensor_vector->batch_sizes)[index],
              (tensor_vector->frames)[index], (tensor_vector->channels)[index], (tensor_vector->widths)[index],
              (tensor_vector->heights)[index]);
}

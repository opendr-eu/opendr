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

#include "object_tracking_2d_deep_sort.h"
#include "target.h"

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/core/core_c.h"

/**
 * Helper function for preprocessing images before feeding them into the deep sort object tracking 2d model.
 * This function follows the OpenDR's object tracking 2d deep sort pre-processing pipeline, which includes the following:
 * a) resizing the image into modelInputSizes[1] x modelInputSizes[0] pixels
 * and b) normalizing the resulting values using meanValues and stdValues
 * @param image image to be preprocesses
 * @param normalizedImg pre-processed data in a matrix
 * @param modelInputSizes size of the center crop (equals the size that the DL model expects)
 * @param meanValues value used for centering the input image
 * @param stdValues value used for scaling the input image
 */
void preprocessDeepSort(cv::Mat *image, cv::Mat *normalizedImg, int modelInputSizes[2], float meanValues[3],
                        float stdValues[3]) {
  // Convert to RGB
  cv::Mat resizedImage;
  cv::cvtColor(*image, resizedImage, cv::COLOR_BGR2RGB);

  // Resize
  cv::resize(resizedImage, resizedImage, cv::Size(modelInputSizes[1], modelInputSizes[0]));

  // Unfold the image into the appropriate format
  // Scale to 0...1
  resizedImage.convertTo(*normalizedImg, CV_32FC3, (1 / 255.0));

  // Normalize
  cv::Scalar meanValue(meanValues[0], meanValues[1], meanValues[2]);
  cv::Scalar stdValue(stdValues[0], stdValues[1], stdValues[2]);

  cv::add(*normalizedImg, meanValue, *normalizedImg);
  cv::multiply(*normalizedImg, stdValue, *normalizedImg);
}

void loadDeepSortModel(const char *modelPath, DeepSortModelT *model) {
  // Initialize model
  model->onnxSession = model->env = model->sessionOptions = NULL;

  // Parse the model JSON file
  std::string basePath(modelPath);
  std::size_t splitPosition = basePath.find_last_of("/");
  splitPosition = splitPosition > 0 ? splitPosition + 1 : 0;
  std::string modelName = basePath.substr(splitPosition);
  std::string modelJsonPath = basePath + "/" + modelName + ".json";
  std::ifstream inStream(modelJsonPath);
  if (!inStream.is_open()) {
    std::cerr << "Cannot open JSON model file" << std::endl;
    return;
  }
  std::string str((std::istreambuf_iterator<char>(inStream)), std::istreambuf_iterator<char>());
  const char *json = str.c_str();

  // Parse JSON
  std::string onnxModelName = jsonGetStringFromKey(json, "model_paths", 0);
  std::string onnxModelPath = basePath + "/" + onnxModelName;
  std::string modelFormat = jsonGetStringFromKey(json, "format", 0);
  // Proceed only if the model is in onnx format
  if (modelFormat != "onnx") {
    std::cerr << "Model not in ONNX format." << std::endl;
    return;
  }

  Ort::Env *env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OpenDR_env");
  Ort::SessionOptions *sessionOptions = new Ort::SessionOptions;
  sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::Session *session = new Ort::Session(*env, onnxModelPath.c_str(), *sessionOptions);
  model->env = env;
  model->onnxSession = session;
  model->sessionOptions = sessionOptions;

  // Should we pass these parameters through the model json file?
  model->meanValue[0] = -0.485f;
  model->meanValue[1] = -0.456f;
  model->meanValue[2] = -0.406f;

  model->stdValue[0] = (1.0f / 0.229f);
  model->stdValue[1] = (1.0f / 0.224f);
  model->stdValue[2] = (1.0f / 0.225f);

  model->modelSize[0] = 64;
  model->modelSize[1] = 128;

  model->batchSize = 1;
  model->inChannels = 3;

  model->features = 512;
}

void freeDeepSortModel(DeepSortModelT *model) {
  if (model->onnxSession) {
    Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);
    delete session;
  }

  if (model->sessionOptions) {
    Ort::SessionOptions *sessionOptions = static_cast<Ort::SessionOptions *>(model->sessionOptions);
    delete sessionOptions;
  }

  if (model->env) {
    Ort::Env *env = static_cast<Ort::Env *>(model->env);
    delete env;
  }
}

void ffDeepSort(DeepSortModelT *model, OpenDRTensorT *tensor, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize = model->batchSize * model->inChannels * model->modelSize[1] * model->modelSize[0];

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {tensor->batchSize, tensor->channels, tensor->width, tensor->height};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"data"};
  std::vector<const char *> outputNodeNames = {"output"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, tensor->data, inputTensorSize, inputNodeDims.data(), 4);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
  assert(outputTensors.size() == 1);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    int tensorSizes[5] = {1, 1, 1, model->batchSize, model->features};
    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
    outputTensorValues->push_back(outputMat);
  }
}

void initRandomOpenDRTensorDs(OpenDRTensorT *tensor, DeepSortModelT *model) {
  int inputTensorSize = 1 * model->batchSize * model->inChannels * model->modelSize[1] * model->modelSize[0];
  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  // Dims of input of model
  loadTensor(tensor, static_cast<void *>(data), model->batchSize, 1, model->inChannels, model->modelSize[1],
             model->modelSize[0]);
  free(data);
}

void forwardDeepSort(DeepSortModelT *model, OpenDRTensorT *tensor, OpenDRTensorVectorT *vector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ffDeepSort(model, tensor, &outputTensorValues);

  int nTensors = static_cast<int>(outputTensorValues.size());
  if (nTensors > 0) {
    int batchSizes[nTensors];
    int frames[nTensors];
    int channels[nTensors];
    int widths[nTensors];
    int heights[nTensors];

    std::vector<OpenDRTensor> tempTensorsVector;
    OpenDRTensorT tempTensor[nTensors];

    for (int i = 0; i < nTensors; i++) {
      initTensor(&(tempTensor[i]));

      batchSizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = model->batchSize;
      heights[i] = model->features;

      loadTensor(&(tempTensor[i]), outputTensorValues[i].ptr<void>(0), batchSizes[i], frames[i], channels[i], widths[i],
                 heights[i]);
      tempTensorsVector.push_back(tempTensor[i]);
    }
    loadTensorVector(vector, tempTensorsVector.data(), nTensors);
    for (int i = 0; i < nTensors; i++) {
      freeTensor(&(tempTensor[i]));
    }
  } else {
    initTensorVector(vector);
  }
}

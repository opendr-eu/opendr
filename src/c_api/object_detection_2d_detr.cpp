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

#include "object_detection_2d_detr.h"
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
 * Helper function for preprocessing images before feeding them into the detr object detection model.
 * This function follows the OpenDR's object detection detr  pre-processing pipeline, which includes the following:
 * a) resizing the image into modelInputSize x modelInputSize pixels
 * and b) normalizing the resulting values using
 * meanValue and stdValue
 * @param image image to be preprocesses
 * @param normalizedImage pre-processed data in a flattened vector
 * @param modelInputSize size of the center crop (equals the size that the DL model expects)
 * @param meanValue values used for centering the input image
 * @param stdValues values used for scaling the input image
 */
void preprocessDetr(cv::Mat *image, cv::Mat *normalizedImage, int modelInputSize, float meanValues[3], float stdValues[3]) {
  // Convert to RGB
  cv::Mat resizedImage;
  cv::cvtColor(*image, resizedImage, cv::COLOR_BGR2RGB);

  // Resize and then get a center crop
  cv::resize(resizedImage, resizedImage, cv::Size(modelInputSize, modelInputSize));

  // Scale to 0...1
  resizedImage.convertTo(*normalizedImage, CV_32FC3, (1 / 255.0));

  cv::Scalar meanValue(meanValues[0], meanValues[1], meanValues[2]);
  cv::Scalar stdValue(stdValues[0], stdValues[1], stdValues[2]);

  cv::add(*normalizedImage, meanValue, *normalizedImage);
  cv::multiply(*normalizedImage, stdValue, *normalizedImage);
}

void loadDetrModel(const char *modelPath, DetrModelT *model) {
  // Initialize model
  model->onnxSession = model->env = model->sessionOptions = NULL;
  model->threshold = 0;

  // Parse the model JSON file
  std::string basePath(modelPath);
  std::size_t splitPos = basePath.find_last_of("/");
  splitPos = splitPos > 0 ? splitPos + 1 : 0;
  std::string modelJsonPath = basePath + "/" + basePath.substr(splitPos) + ".json";

  std::ifstream inStream(modelJsonPath);
  if (!inStream.is_open()) {
    std::cerr << "Cannot open JSON model file." << std::endl;
    return;
  }
  std::string str((std::istreambuf_iterator<char>(inStream)), std::istreambuf_iterator<char>());
  const char *json = str.c_str();

  // Parse JSON
  std::string modelPaths = jsonGetStringFromKey(json, "model_paths", 0);
  std::string onnxModelPath = basePath + "/" + modelPaths;
  std::string modelFormat = jsonGetStringFromKey(json, "format", 0);

  // Parse inference params
  float threshold = jsonGetFloatFromKeyInInferenceParams(json, "threshold", 0);
  model->threshold = threshold;

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

  model->meanValue[0] = -0.485f;
  model->meanValue[1] = -0.456f;
  model->meanValue[2] = -0.406f;

  model->stdValue[0] = 0.229f;
  model->stdValue[1] = 0.224f;
  model->stdValue[2] = 0.225f;

  model->modelSize = 800;

  model->features = 100;
  model->outputSizes[0] = 92;
  model->outputSizes[1] = 4;
}

void freeDetrModel(DetrModelT *model) {
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

void ffDetr(DetrModelT *model, OpenDRTensorT *tensor, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data for preprocessing
  size_t inputTensorSize = model->modelSize * model->modelSize * 3;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {tensor->batchSize, tensor->channels, tensor->width, tensor->height};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"data"};
  std::vector<const char *> outputNodeNames = {"pred_logits", "pred_boxes"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, tensor->data, inputTensorSize, inputNodeDims.data(), 4);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 2);
  assert(outputTensors.size() == 2);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    int tensorSizes[5] = {1, 1, 1, model->features, model->outputSizes[i]};

    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
    outputTensorValues->push_back(outputMat);
  }
}

void initRandomOpenDRTensorDetr(OpenDRTensorT *tensor, DetrModelT *model) {
  // Prepare the input data with random values
  int inputTensorSize = model->modelSize * model->modelSize * 3;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  // change data structure so channels are the last iterable dimension
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  loadTensor(tensor, static_cast<void *>(data), 1, 1, 3, model->modelSize, model->modelSize);
  free(data);
}

void forwardDetr(DetrModelT *model, OpenDRTensorT *tensor, OpenDRTensorVectorT *vector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ffDetr(model, tensor, &outputTensorValues);

  int nTensors = static_cast<int>(outputTensorValues.size());
  if (nTensors > 0) {
    int batchSizes[nTensors];
    int frames[nTensors];
    int channels[nTensors];
    int widths[nTensors];
    int heights[nTensors];

    std::vector<OpenDRTensor> tempTensorsVector;
    OpenDRTensorT tempTensors[nTensors];

    for (int i = 0; i < nTensors; i++) {
      initTensor(&(tempTensors[i]));

      batchSizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = 1;
      if (i == 0) {
        heights[i] = model->outputSizes[0];
      } else {
        heights[i] = model->outputSizes[1];
      }
      loadTensor(&(tempTensors[i]), outputTensorValues[i].ptr<void>(0), batchSizes[i], frames[i], channels[i], widths[i],
                 heights[i]);
      tempTensorsVector.push_back(tempTensors[i]);
    }
    loadTensorVector(vector, tempTensorsVector.data(), nTensors);
    for (int i = 0; i < nTensors; i++) {
      freeTensor(&(tempTensors[i]));
    }

  } else {
    initTensorVector(vector);
  }
}

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

#include "activity_recognition_x3d.h"
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
 * Helper function for preprocessing images before feeding them into the activity recognition x3d model.
 * This function follows the OpenDR's activity recognition x3d pre-processing pipeline, which includes the following:
 * a) resizing the image into modelInputSize x modelInputSize pixels relative to the original ratio,
 * b) normalizing the resulting values using meanValue.
 * @param image image to be preprocesses
 * @param normalizedImage pre-processed data in a matrix
 * @param modelInputSize size of the center crop (equals the size that the DL model expects)
 * @param meanValue value used for centering the input image
 * @param stdValue value used for scaling the input image
 */
void preprocessX3d(cv::Mat *image, cv::Mat *normalizedImage, int modelInputSize, float meanValue, float stdValue) {
  // Convert to RGB
  cv::Mat imageRgb;
  cv::cvtColor(*image, imageRgb, cv::COLOR_BGR2RGB);

  // Resize with ratio
  double scale = (static_cast<double>(modelInputSize) / static_cast<double>(imageRgb.rows));
  cv::resize(imageRgb, imageRgb, cv::Size(), scale, scale);

  // Convert to 32f and normalize
  imageRgb.convertTo(*normalizedImage, CV_32FC3, stdValue, meanValue);
}

void loadX3dModel(const char *modelPath, char *mode, X3dModelT *model) {
  // Initialize model
  model->onnxSession = model->env = model->sessionOptions = NULL;

  Ort::Env *env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "opendr_env");
  Ort::SessionOptions *sessionOptions = new Ort::SessionOptions;
  sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::Session *session = new Ort::Session(*env, modelPath, *sessionOptions);
  model->env = env;
  model->onnxSession = session;
  model->sessionOptions = sessionOptions;

  // Should we pass these parameters through the model json file?
  model->meanValue = -128.0f / 255.0f;
  model->imgScale = (1.0f / 255.0f);

  std::string modeName = mode;
  if (modeName == "l") {
    model->modelSize = 312;
    model->framesPerClip = 16;
  } else if (modeName == "m") {
    model->modelSize = 224;
    model->framesPerClip = 16;
  } else if (modeName == "s") {
    model->modelSize = 160;
    model->framesPerClip = 13;
  } else if (modeName == "xs") {
    model->modelSize = 160;
    model->framesPerClip = 4;
  } else {
    std::cout << "mode: {'" << modeName
              << "'} is not a compatible choice, please use one of {'xs', 's', 'm', 'l'} and try again." << std::endl;
    return;
  }

  model->batchSize = 1;
  model->inChannels = 3;

  model->features = 400;
}

void freeX3dModel(X3dModelT *model) {
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

void ffX3d(X3dModelT *model, OpendrTensorT *tensor, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize = model->batchSize * model->inChannels * model->framesPerClip * model->modelSize * model->modelSize;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {tensor->batchSize, tensor->channels, tensor->frames, tensor->width, tensor->height};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"video"};
  std::vector<const char *> outputNodeNames = {"classes"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, tensor->data, inputTensorSize, inputNodeDims.data(), 5);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
  assert(outputTensors.size() == 1);

  // Get the results back
  float *tensorData = outputTensors.front().GetTensorMutableData<float>();

  int tensorSizes[5] = {1, 1, 1, model->batchSize, model->features};

  cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
  outputTensorValues->push_back(outputMat);
}

void initRandomOpendrTensorX3d(OpendrTensorT *tensor, X3dModelT *model) {
  // Prepare the input dimensions
  // Dims of input data
  int inputTensorSize = model->batchSize * model->framesPerClip * model->inChannels * model->modelSize * model->modelSize;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  // change data structure so channels are the last iterable dimension
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  // Dims of input of model
  loadTensor(tensor, static_cast<void *>(data), model->batchSize, model->framesPerClip, model->inChannels, model->modelSize,
             model->modelSize);
  free(data);
}

void forwardX3d(X3dModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ffX3d(model, tensor, &outputTensorValues);

  int nTensors = static_cast<int>(outputTensorValues.size());
  if (nTensors > 0) {
    int batchSizes[nTensors];
    int frames[nTensors];
    int channels[nTensors];
    int widths[nTensors];
    int heights[nTensors];

    // TODO: use std::unique_ptr to not have to free after the copy
    std::vector<OpendrTensor> tempTensorsVector;
    OpendrTensorT tempTensors[nTensors];

    for (int i = 0; i < nTensors; i++) {
      initTensor(&(tempTensors[i]));

      batchSizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = model->batchSize;
      heights[i] = model->features;

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

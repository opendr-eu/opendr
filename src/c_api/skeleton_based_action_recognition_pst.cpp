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

#include "skeleton_based_action_recognition_pst.h"
#include "target.h"

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
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

void loadPstModel(const char *modelPath, PstModelT *model) {
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
  model->batchSize = 128;
  model->inChannels = 2;
  model->features = 300;
  model->nPoint = 18;  // same as the output of open pose
  model->nPerson = 2;

  model->nClasses = 60;
}

void freePstModel(PstModelT *model) {
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

void ffPst(PstModelT *model, OpendrTensorT *tensor, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize = model->batchSize * model->inChannels * model->features * model->nPoint * model->nPerson;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {tensor->batchSize, tensor->frames, tensor->channels, tensor->width, tensor->height};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"onnx_input"};
  std::vector<const char *> outputNodeNames = {"onnx_output"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, tensor->data, inputTensorSize, inputNodeDims.data(), 5);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
  assert(outputTensors.size() == 1);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    int tensorSizes[5] = {1, 1, 1, model->batchSize, model->nClasses};

    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
    outputTensorValues->push_back(outputMat);
  }
}

void initRandomOpendrTensorPst(OpendrTensorT *tensor, PstModelT *model) {
  int inputTensorSize = model->batchSize * model->inChannels * model->features * model->nPoint * model->nPerson;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  loadTensor(tensor, static_cast<void *>(data), model->batchSize, model->inChannels, model->features, model->nPoint,
             model->nPerson);
  free(data);
}

void forwardPst(PstModelT *model, OpendrTensorT *tensor, OpendrTensorVectorT *vector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ffPst(model, tensor, &outputTensorValues);

  int nTensors = static_cast<int>(outputTensorValues.size());
  if (nTensors > 0) {
    int batchSizes[nTensors];
    int frames[nTensors];
    int channels[nTensors];
    int widths[nTensors];
    int heights[nTensors];

    std::vector<OpendrTensor> tempTensorsVector;
    OpendrTensorT tempTensors[nTensors];

    for (int i = 0; i < nTensors; i++) {
      initTensor(&(tempTensors[i]));

      batchSizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = model->batchSize;
      heights[i] = model->nClasses;

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

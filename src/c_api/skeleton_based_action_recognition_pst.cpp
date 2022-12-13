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

#include "skeleton_based_action_recognition_pst.h"
#include "target.h"

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
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

void load_pst_model(const char *modelPath, pst_model_t *model) {
  // Initialize model
  model->onnx_session = model->env = model->session_options = NULL;

  Ort::Env *env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "opendr_env");
  Ort::SessionOptions *sessionOptions = new Ort::SessionOptions;
  sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::Session *session = new Ort::Session(*env, modelPath, *sessionOptions);
  model->env = env;
  model->onnx_session = session;
  model->session_options = sessionOptions;

  // Should we pass these parameters through the model json file?
  model->batch_size = 128;
  model->in_channels = 2;
  model->features = 300;
  model->num_point = 18;  // same as the output of openpose
  model->num_person = 2;

  model->num_classes = 60;
}

void free_pst_model(pst_model_t *model) {
  if (model->onnx_session) {
    Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);
    delete session;
  }

  if (model->session_options) {
    Ort::SessionOptions *sessionOptions = static_cast<Ort::SessionOptions *>(model->session_options);
    delete sessionOptions;
  }

  if (model->env) {
    Ort::Env *env = static_cast<Ort::Env *>(model->env);
    delete env;
  }
}

void ff_pst(pst_model_t *model, opendr_tensor_t *inputTensorValues, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize = model->batch_size * model->in_channels * model->features * model->num_point * model->num_person;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {model->batch_size, model->in_channels, model->features, model->num_point,
                                        model->num_person};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"onnx_input"};
  std::vector<const char *> outputNodeNames = {"onnx_output"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor =
    Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues->data, inputTensorSize, inputNodeDims.data(), 5);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
  assert(outputTensors.size() == 1);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    int tensorSizes[5] = {1, 1, 1, model->batch_size, model->num_classes};

    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
    outputTensorValues->push_back(outputMat);
  }
}

void init_random_opendr_tensor_pst(opendr_tensor_t *inputTensorValues, pst_model_t *model) {
  int inputTensorSize = model->batch_size * model->in_channels * model->features * model->num_point * model->num_person;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  load_tensor(inputTensorValues, static_cast<void *>(data), model->batch_size, model->in_channels, model->features,
              model->num_point, model->num_person);
  free(data);
}

void forward_pst(pst_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ff_pst(model, inputTensorValues, &outputTensorValues);

  int nTensors = static_cast<int>(outputTensorValues.size());
  if (nTensors > 0) {
    int batch_sizes[nTensors];
    int frames[nTensors];
    int channels[nTensors];
    int widths[nTensors];
    int heights[nTensors];

    std::vector<opendr_tensor> tempTensorsVector;
    opendr_tensor_t tempTensors[nTensors];

    for (int i = 0; i < nTensors; i++) {
      init_tensor(&(tempTensors[i]));

      batch_sizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = model->batch_size;
      heights[i] = model->num_classes;

      load_tensor(&(tempTensors[i]), outputTensorValues[i].ptr<void>(0), batch_sizes[i], frames[i], channels[i], widths[i],
                  heights[i]);
      tempTensorsVector.push_back(tempTensors[i]);
    }
    load_tensor_vector(tensorVector, tempTensorsVector.data(), nTensors);
    for (int i = 0; i < nTensors; i++) {
      free_tensor(&(tempTensors[i]));
    }
  } else {
    init_tensor_vector(tensorVector);
  }
}

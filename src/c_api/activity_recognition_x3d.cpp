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

#include "activity_recognition_x3d.h"
#include "target.h"

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <boost/filesystem.hpp>
#include <cmath>
#include <cstring>
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

/**
 * Helper function for preprocessing images before feeding them into the lightweight open pose estimator model.
 * This function follows the OpenDR's lightweight open pose  pre-processing pipeline, which includes the following:
 * a) resizing the image into modelInputSize x modelInputSize pixels relative to the original ratio,
 * b) normalizing the resulting values using meanValue and c) padding image into a standard size.
 * @param image image to be preprocesses
 * @param normalizedImage pre-processed data in a matrix
 * @param modelInputSize size of the center crop (equals the size that the DL model expects)
 * @param meanValue value used for centering the input image
 * @param imageScale value used for scaling the input image
 */
void preprocess_x3d(cv::Mat *image, cv::Mat *normalizedImage, int modelInputSize, float meanValue, float imageScale) {
  // Convert to RGB
  cv::Mat imageRgb;
  cv::cvtColor(*image, imageRgb, cv::COLOR_BGR2RGB);

  // Resize with ratio
  double scale = (static_cast<double>(modelInputSize) / static_cast<double>(imageRgb.rows));
  cv::resize(imageRgb, imageRgb, cv::Size(), scale, scale);

  // Convert to 32f and normalize
  imageRgb.convertTo(*normalizedImage, CV_32FC3, imageScale, meanValue);
}

void load_x3d_model(const char *modelPath, char *mode, x3d_model_t *model) {
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
  model->mean_value = -128.0f / 255.0f;
  model->img_scale = (1.0f / 255.0f);

  //  std::string model_name = "l";
  std::string modeName = mode;
  if (modeName == "l") {
    model->model_size = 312;
    model->frames_per_clip = 16;
  } else if (modeName == "m") {
    model->model_size = 224;
    model->frames_per_clip = 16;
  } else if (modeName == "s") {
    model->model_size = 160;
    model->frames_per_clip = 13;
  } else {
    model->model_size = 160;
    model->frames_per_clip = 4;
  }

  model->batch_size = 1;
  model->in_channels = 3;

  model->features = 400;
}

void free_x3d_model(x3d_model_t *model) {
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

void ff_x3d(x3d_model_t *model, opendr_tensor_t *inputTensorValues, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize =
    model->batch_size * model->in_channels * model->frames_per_clip * model->model_size * model->model_size;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {model->batch_size, model->in_channels, model->frames_per_clip, model->model_size,
                                        model->model_size};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"video"};
  std::vector<const char *> outputNodeNames = {"classes"};

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
  float *tensorData = outputTensors.front().GetTensorMutableData<float>();

  int tensorSizes[5] = {1, 1, 1, model->batch_size, model->features};

  cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
  outputTensorValues->push_back(outputMat);
}

void init_random_opendr_tensor_x3d(opendr_tensor_t *inputTensorValues, x3d_model_t *model) {
  // Prepare the input dimensions
  // Dims of input data
  int inputTensorSize = model->batch_size * model->frames_per_clip * model->in_channels * model->model_size * model->model_size;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  // change data structure so channels are the last iterable dimension
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  // Dims of input of model
  load_tensor(inputTensorValues, static_cast<void *>(data), model->batch_size, model->frames_per_clip, model->in_channels,
              model->model_size, model->model_size);
  free(data);
}

void forward_x3d(x3d_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ff_x3d(model, inputTensorValues, &outputTensorValues);

  int nTensors = static_cast<int>(outputTensorValues.size());
  if (nTensors > 0) {
    int batchSizes[nTensors];
    int frames[nTensors];
    int channels[nTensors];
    int widths[nTensors];
    int heights[nTensors];

    std::vector<opendr_tensor> tempTensorsVector;
    opendr_tensor_t tempTensors[nTensors];

    for (int i = 0; i < nTensors; i++) {
      batchSizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = model->batch_size;
      heights[i] = model->features;

      load_tensor(&(tempTensors[i]), outputTensorValues[i].ptr<void>(0), batchSizes[i], frames[i], channels[i], widths[i],
                  heights[i]);
      tempTensorsVector.push_back(tempTensors[i]);
    }
    load_tensor_vector(tensorVector, tempTensorsVector.data(), nTensors);
    for (int i = 0; i < nTensors; i++) {
      free_tensor(&(tempTensors[i]));
    }

  } else {
    initialize_tensor_vector(tensorVector);
  }
}

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
 * a) resizing the image into modelInputSizes[1] x modelInputSizes[0] pixels and b) normalizing the resulting values using
 * meanValues and stdValues
 * @param image image to be preprocesses
 * @param normalizedImg pre-processed data in a flattened vector
 * @param modelInputSizes size of the center crop (equals the size that the DL model expects)
 * @param meanValues value used for centering the input image
 * @param stdValues value used for scaling the input image
 */
void preprocess_deep_sort(cv::Mat *image, cv::Mat *normalizedImg, int modelInputSizes[2], float meanValues[3],
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

void load_deep_sort_model(const char *modelPath, deep_sort_model_t *model) {
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
  model->mean_value[0] = -0.485f;
  model->mean_value[1] = -0.456f;
  model->mean_value[2] = -0.406f;

  model->std_value[0] = (1.0f / 0.229f);
  model->std_value[1] = (1.0f / 0.224f);
  model->std_value[2] = (1.0f / 0.225f);

  model->model_size[0] = 64;
  model->model_size[1] = 128;

  model->batch_size = 1;
  model->in_channels = 3;

  model->features = 512;
}

void free_deep_sort_model(deep_sort_model_t *model) {
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

void ff_deep_sort(deep_sort_model_t *model, opendr_tensor_t *inputTensorValues, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize = model->batch_size * model->in_channels * model->model_size[1] * model->model_size[0];

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {model->batch_size, model->in_channels, model->model_size[1], model->model_size[0]};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"data"};
  std::vector<const char *> outputNodeNames = {"output"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor =
    Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues->data, inputTensorSize, inputNodeDims.data(), 4);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
  assert(outputTensors.size() == 1);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    int tensorSizes[5] = {1, 1, 1, model->batch_size, model->features};
    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
    outputTensorValues->push_back(outputMat);
  }
}

void init_random_opendr_tensor_ds(opendr_tensor_t *inputTensorValues, deep_sort_model_t *model) {
  int inputTensorSize = 1 * model->batch_size * model->in_channels * model->model_size[1] * model->model_size[0];
  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  // Dims of input of model
  load_tensor(inputTensorValues, static_cast<void *>(data), 1, model->batch_size, model->in_channels, model->model_size[1],
              model->model_size[0]);
  free(data);
}

void forward_deep_sort(deep_sort_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ff_deep_sort(model, inputTensorValues, &outputTensorValues);

  int nTensors = static_cast<int>(outputTensorValues.size());
  if (nTensors > 0) {
    int batch_sizes[nTensors];
    int frames[nTensors];
    int channels[nTensors];
    int widths[nTensors];
    int heights[nTensors];

    std::vector<opendr_tensor> temp_tensors;
    opendr_tensor_t temp_tensor[nTensors];

    for (int i = 0; i < nTensors; i++) {
      init_tensor(&(temp_tensor[i]));

      batch_sizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = model->batch_size;
      heights[i] = model->features;

      load_tensor(&(temp_tensor[i]), outputTensorValues[i].ptr<void>(0), batch_sizes[i], frames[i], channels[i], widths[i],
                  heights[i]);
      temp_tensors.push_back(temp_tensor[i]);
    }
    load_tensor_vector(tensorVector, temp_tensors.data(), nTensors);
    for (int i = 0; i < nTensors; i++) {
      free_tensor(&(temp_tensor[i]));
    }
  } else {
    init_tensor_vector(tensorVector);
  }
}

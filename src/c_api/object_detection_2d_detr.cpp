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

#include "object_detection_2d_detr.h"
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
 * Helper function for preprocessing images before feeding them into the detr object detection model.
 * This function follows the OpenDR's object detection detr  pre-processing pipeline, which includes the following:
 * a) resizing the image into modelInputSize x modelInputSize pixels and b) normalizing the resulting values using
 * meanValue and stdValue
 * @param image image to be preprocesses
 * @param data pre-processed data in a flattened vector
 * @param modelInputSize size of the center crop (equals the size that the DL model expects)
 * @param meanValue values used for centering the input image
 * @param stdValues values used for scaling the input image
 */
void preprocess_detr(cv::Mat *image, cv::Mat *normalizedImage, int modelInputSize, float meanValues[3], float stdValues[3]) {
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

void load_detr_model(const char *modelPath, detr_model_t *model) {
  // Initialize model
  model->onnx_session = model->env = model->session_options = NULL;
  model->threshold = 0;

  // Parse the model JSON file
  std::string modelJsonPath(modelPath);
  std::size_t splitPos = modelJsonPath.find_last_of("/");
  splitPos = splitPos > 0 ? splitPos + 1 : 0;
  std::string basePath = modelJsonPath;
  modelJsonPath = basePath + "/" + modelJsonPath.substr(splitPos) + ".json";

  std::ifstream in_stream(modelJsonPath);
  if (!in_stream.is_open()) {
    std::cerr << "Cannot open JSON model file" << std::endl;
    return;
  }
  std::string str((std::istreambuf_iterator<char>(in_stream)), std::istreambuf_iterator<char>());
  const char *json = str.c_str();

  // Parse JSON
  std::string modelPaths = json_get_key_string(json, "model_paths", 0);
  std::string onnxModelPath = basePath + "/" + modelPaths;
  std::string modelFormat = json_get_key_string(json, "format", 0);

  // Parse inference params
  float threshold = json_get_key_from_inference_params(json, "threshold", 0);
  model->threshold = threshold;

  // Proceed only if the model is in onnx format
  if (modelFormat != "onnx") {
    std::cerr << "Model not in ONNX format." << std::endl;
    return;
  }

  Ort::Env *env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "opendr_env");
  Ort::SessionOptions *sessionOptions = new Ort::SessionOptions;
  sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  Ort::Session *session = new Ort::Session(*env, onnxModelPath.c_str(), *sessionOptions);
  model->env = env;
  model->onnx_session = session;
  model->session_options = sessionOptions;

  // Should we pass these parameters through the model json file?
  model->mean_value[0] = -0.485f;
  model->mean_value[1] = -0.456f;
  model->mean_value[2] = -0.406f;

  model->std_value[0] = 0.229f;
  model->std_value[1] = 0.224f;
  model->std_value[2] = 0.225f;

  model->model_size = 800;

  model->features = 100;
  model->output_sizes[0] = 92;
  model->output_sizes[1] = 4;
}

void free_detr_model(detr_model_t *model) {
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

void ff_detr(detr_model_t *model, opendr_tensor_t *inputTensorValues, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data for preprocessing
  size_t inputTensorSize = model->model_size * model->model_size * 3;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {inputTensorValues->batch_size, inputTensorValues->channels, inputTensorValues->width,
                                        inputTensorValues->height};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"data"};
  std::vector<const char *> outputNodeNames = {"pred_logits", "pred_boxes"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor =
    Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues->data, inputTensorSize, inputNodeDims.data(), 4);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 2);
  assert(outputTensors.size() == 2);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    int tensorSizes[5] = {1, 1, 1, model->features, model->output_sizes[i]};

    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
    outputTensorValues->push_back(outputMat);
  }
}

void init_random_opendr_tensor_detr(opendr_tensor_t *inputTensorValues, detr_model_t *model) {
  // Prepare the input data with random values
  int inputTensorSize = model->model_size * model->model_size * 3;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  // change data structure so channels are the last iterable dimension
  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  load_tensor(inputTensorValues, static_cast<void *>(data), 1, 1, 3, model->model_size, model->model_size);
  free(data);
}

void forward_detr(detr_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ff_detr(model, inputTensorValues, &outputTensorValues);

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
      init_tensor(&(tempTensors[i]));

      batchSizes[i] = 1;
      frames[i] = 1;
      channels[i] = 1;
      widths[i] = 1;
      if (i == 0) {
        heights[i] = model->output_sizes[0];
      } else {
        heights[i] = model->output_sizes[1];
      }
      load_tensor(&(tempTensors[i]), outputTensorValues[i].ptr<void>(0), batchSizes[i], frames[i], channels[i], widths[i],
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

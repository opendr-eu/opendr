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

#include "lightweight_open_pose.h"
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

/**
 * Helper function for preprocessing images before feeding them into the lightweight open pose estimator model.
 * This function follows the OpenDR's lightweight open pose  pre-processing pipeline, which includes the following:
 * a) resizing the image into modelInputSize x modelInputSize pixels relative to the original ratio,
 * b) normalizing the resulting values using meanValue and c) padding image into a standard size.
 * @param image image to be preprocesses
 * @param preprocessedImage opencv Mat that pre-processed data will be saved
 * @param modelInputSize size of the center crop (equals the size that the DL model expects)
 * @param meanValue value used for centering the input image
 * @param imgScale value used for scaling the input image
 */
void preprocess_open_pose(cv::Mat *image, cv::Mat *preprocessedImage, int modelInputSize, float meanValue, float imgScale) {
  // Convert to RGB
  cv::Mat resizedImage;
  cv::cvtColor(*image, resizedImage, cv::COLOR_BGR2RGB);
  cv::cvtColor(resizedImage, resizedImage, cv::COLOR_RGB2BGR);

  // Resize and then get a center crop
  double scale = (static_cast<double>(modelInputSize) / static_cast<double>(resizedImage.rows));
  cv::resize(resizedImage, resizedImage, cv::Size(), scale, scale);

  // Convert to float32 and normalize
  cv::Mat normalizedImage;
  resizedImage.convertTo(normalizedImage, CV_32FC3, imgScale, meanValue);

  // Padding
  int h = normalizedImage.rows;
  int w = normalizedImage.cols;

  const float stride = 8.0f;
  int maxWidth = std::max(modelInputSize, w);
  cv::Size minDims = cv::Size(maxWidth, modelInputSize);

  h = std::min(h, minDims.height);
  minDims.height = ceil((minDims.height / stride)) * stride;

  minDims.width = std::max(minDims.width, w);
  minDims.width = ceil((minDims.width / stride)) * stride;

  int pad[4];
  pad[0] = static_cast<int>((minDims.height - h) / 2);
  pad[1] = static_cast<int>((minDims.width - w) / 2);
  pad[2] = minDims.height - h - pad[0];
  pad[3] = minDims.width - w - pad[1];

  cv::Scalar padValue(0, 0, 0);
  cv::copyMakeBorder(normalizedImage, *preprocessedImage, pad[0], pad[2], pad[1], pad[3], cv::BORDER_CONSTANT, padValue);
}

void load_open_pose_model(const char *modelPath, open_pose_model_t *model) {
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
  model->mean_value = -128.0f / 256.0f;
  model->img_scale = (1.0f / 256.0f);
  model->model_size = 256;

  model->num_refinement_stages = 2;
  model->output_size = (model->num_refinement_stages + 1) * 2;

  model->even_channel_output = 38;
  model->odd_channel_output = 19;
  model->stride = 0;
  model->batch_size = 1;
  if (model->stride == 0) {
    model->width_output = 32;
    model->height_output = 49;
  } else {
    model->width_output = 16;
    model->height_output = 35;
  }
}

void free_open_pose_model(open_pose_model_t *model) {
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

void ff_open_pose(open_pose_model_t *model, opendr_tensor_t *inputTensorValues, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize = model->model_size * model->model_size * 3;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {inputTensorValues->batch_size, inputTensorValues->channels, inputTensorValues->width,
                                        inputTensorValues->height};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"data"};
  std::vector<const char *> outputNodeNames = {"stage_0_output_1_heatmaps", "stage_0_output_0_pafs"};
  if (model->num_refinement_stages == 2) {
    outputNodeNames.push_back("stage_1_output_1_heatmaps");
    outputNodeNames.push_back("stage_1_output_0_pafs");
    outputNodeNames.push_back("stage_2_output_1_heatmaps");
    outputNodeNames.push_back("stage_2_output_0_pafs");
  }
  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor =
    Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues->data, inputTensorSize, inputNodeDims.data(), 4);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), model->output_size);
  assert(outputTensors.size() == model->output_size);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensor_data = outputTensors[i].GetTensorMutableData<float>();

    int channelDim;
    if ((i % 2) == 0) {
      channelDim = model->even_channel_output;
    } else {
      channelDim = model->odd_channel_output;
    }

    int tensorSizes[5] = {1, model->batch_size, channelDim, model->width_output, model->height_output};

    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensor_data));
    outputTensorValues->push_back(outputMat);
  }
}

void init_random_opendr_tensor_op(opendr_tensor_t *inputTensorValues, open_pose_model_t *model) {
  int inputTensorSize = model->model_size * model->model_size * 3;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));

  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  load_tensor(inputTensorValues, static_cast<void *>(data), 1, 1, 3, model->model_size, model->model_size);
  free(data);
}

void init_opendr_tensor_from_img_op(opendr_image_t *image, opendr_tensor_t *inputTensorValues, open_pose_model_t *model) {
  int inputTensorSize = model->model_size * model->model_size * 3;

  cv::Mat *opencvImage = (static_cast<cv::Mat *>(image->data));
  cv::Mat normImage;
  preprocess_open_pose(opencvImage, &normImage, model->model_size, model->mean_value, model->img_scale);

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  for (unsigned int j = 0; j < model->model_size; ++j) {
    for (unsigned int k = 0; k < model->model_size; ++k) {
      cv::Vec3f currentPixel = normImage.at<cv::Vec3f>(j, k);
      data[0 * model->model_size * model->model_size + j * model->model_size + k] = currentPixel[0];
      data[1 * model->model_size * model->model_size + j * model->model_size + k] = currentPixel[1];
      data[2 * model->model_size * model->model_size + j * model->model_size + k] = currentPixel[2];
    }
  }

  load_tensor(inputTensorValues, static_cast<void *>(data), 1, 1, 3, model->model_size, model->model_size);
  free(data);
}

void forward_open_pose(open_pose_model_t *model, opendr_tensor_t *inputTensorValues, opendr_tensor_vector_t *tensorVector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ff_open_pose(model, inputTensorValues, &outputTensorValues);

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
      if ((i % 2) == 0) {
        channels[i] = model->even_channel_output;
      } else {
        channels[i] = model->odd_channel_output;
      }
      widths[i] = model->width_output;
      heights[i] = model->height_output;

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

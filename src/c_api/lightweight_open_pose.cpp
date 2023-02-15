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

#include "lightweight_open_pose.h"
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

/**
 * Helper function for preprocessing images before feeding them into the lightweight open pose estimator model.
 * This function follows the OpenDR's lightweight open pose  pre-processing pipeline, which includes the following:
 * a) resizing the image into modelInputSize x modelInputSize pixels relative to the original ratio,
 * b) normalizing the resulting values using meanValue
 * and c) padding image into a standard size.
 * @param image image to be preprocesses
 * @param normalizedImg pre-processed data in a matrix
 * @param modelInputSize size of the center crop (equals the size that the DL model expects)
 * @param meanValue value used for centering the input image
 * @param stdValue value used for scaling the input image
 */
void preprocessOpenPose(cv::Mat *image, cv::Mat *normalizedImg, int modelInputSize, float meanValue, float stdValue) {
  // Convert to RGB
  cv::Mat resizedImage;
  cv::cvtColor(*image, resizedImage, cv::COLOR_BGR2RGB);
  cv::cvtColor(resizedImage, resizedImage, cv::COLOR_RGB2BGR);

  // Resize and then get a center crop
  double scale = (static_cast<double>(modelInputSize) / static_cast<double>(resizedImage.rows));
  cv::resize(resizedImage, resizedImage, cv::Size(), scale, scale);

  // Convert to float32 and normalize
  cv::Mat normalizedImage;
  resizedImage.convertTo(normalizedImage, CV_32FC3, stdValue, meanValue);

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
  cv::copyMakeBorder(normalizedImage, *normalizedImg, pad[0], pad[2], pad[1], pad[3], cv::BORDER_CONSTANT, padValue);
}

void loadOpenPoseModel(const char *modelPath, OpenPoseModelT *model) {
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
    std::cerr << "Cannot open JSON model file." << std::endl;
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
  model->meanValue = -128.0f / 256.0f;
  model->imgScale = (1.0f / 256.0f);
  model->modelSize = 256;

  model->nRefinementStages = 2;
  model->outputSize = (model->nRefinementStages + 1) * 2;

  model->evenChannelOutput = 38;
  model->oddChannelOutput = 19;
  model->stride = 0;
  model->batchSize = 1;
  if (model->stride == 0) {
    model->widthOutput = 32;
    model->heightOutput = 49;
  } else {
    model->widthOutput = 16;
    model->heightOutput = 35;
  }
}

void freeOpenPoseModel(OpenPoseModelT *model) {
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

void ffOpenPose(OpenPoseModelT *model, OpenDRTensorT *tensor, std::vector<cv::Mat> *outputTensorValues) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);

  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  // Dims of input data
  size_t inputTensorSize = model->modelSize * model->modelSize * 3;

  // Dims of input of model
  std::vector<int64_t> inputNodeDims = {tensor->batchSize, tensor->channels, tensor->width, tensor->height};

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"data"};
  std::vector<const char *> outputNodeNames = {"stage_0_output_1_heatmaps", "stage_0_output_0_pafs"};
  if (model->nRefinementStages == 2) {
    outputNodeNames.push_back("stage_1_output_1_heatmaps");
    outputNodeNames.push_back("stage_1_output_0_pafs");
    outputNodeNames.push_back("stage_2_output_1_heatmaps");
    outputNodeNames.push_back("stage_2_output_0_pafs");
  }
  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, tensor->data, inputTensorSize, inputNodeDims.data(), 4);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), model->outputSize);
  assert(outputTensors.size() == model->outputSize);

  // Get the results back
  for (int i = 0; i < outputTensors.size(); i++) {
    float *tensorData = outputTensors[i].GetTensorMutableData<float>();

    int channelDim;
    if ((i % 2) == 0) {
      channelDim = model->evenChannelOutput;
    } else {
      channelDim = model->oddChannelOutput;
    }

    int tensorSizes[5] = {1, model->batchSize, channelDim, model->widthOutput, model->heightOutput};

    cv::Mat outputMat(5, tensorSizes, CV_32F, static_cast<void *>(tensorData));
    outputTensorValues->push_back(outputMat);
  }
}

void initRandomOpenDRTensorOp(OpenDRTensorT *tensor, OpenPoseModelT *model) {
  int inputTensorSize = model->modelSize * model->modelSize * 3;

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));

  for (unsigned int j = 0; j < inputTensorSize; ++j) {
    data[j] = (((float)rand() / (RAND_MAX)) * 2) - 1;
  }

  loadTensor(tensor, static_cast<void *>(data), 1, 1, 3, model->modelSize, model->modelSize);
  free(data);
}

void initOpenDRTensorFromImgOp(OpenDRImageT *image, OpenDRTensorT *tensor, OpenPoseModelT *model) {
  int inputTensorSize = model->modelSize * model->modelSize * 3;

  cv::Mat *opencvImage = (static_cast<cv::Mat *>(image->data));
  cv::Mat normImage;
  preprocessOpenPose(opencvImage, &normImage, model->modelSize, model->meanValue, model->imgScale);

  float *data = static_cast<float *>(malloc(inputTensorSize * sizeof(float)));
  for (unsigned int j = 0; j < model->modelSize; ++j) {
    for (unsigned int k = 0; k < model->modelSize; ++k) {
      cv::Vec3f currentPixel = normImage.at<cv::Vec3f>(j, k);
      data[0 * model->modelSize * model->modelSize + j * model->modelSize + k] = currentPixel[0];
      data[1 * model->modelSize * model->modelSize + j * model->modelSize + k] = currentPixel[1];
      data[2 * model->modelSize * model->modelSize + j * model->modelSize + k] = currentPixel[2];
    }
  }

  loadTensor(tensor, static_cast<void *>(data), 1, 1, 3, model->modelSize, model->modelSize);
  free(data);
}

void forwardOpenPose(OpenPoseModelT *model, OpenDRTensorT *tensor, OpenDRTensorVectorT *vector) {
  // Get the feature vector for the current image
  std::vector<cv::Mat> outputTensorValues;
  ffOpenPose(model, tensor, &outputTensorValues);

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
      if ((i % 2) == 0) {
        channels[i] = model->evenChannelOutput;
      } else {
        channels[i] = model->oddChannelOutput;
      }
      widths[i] = model->widthOutput;
      heights[i] = model->heightOutput;

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

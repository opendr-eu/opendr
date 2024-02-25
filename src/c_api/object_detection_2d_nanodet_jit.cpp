// Copyright 2020-2024 OpenDR European Project
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

#include "object_detection_2d_nanodet_jit.h"

#include <document.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

/**
 * Helper class holder of c++ values and jit model.
 */
class NanoDet {
private:
  torch::DeviceType mDevice;
  torch::jit::script::Module mNetwork;
  torch::Tensor mMeanTensor;
  torch::Tensor mStdTensor;
  std::vector<std::string> mLabels;

public:
  NanoDet(torch::jit::script::Module network, torch::Tensor meanValues, torch::Tensor stdValues, torch::DeviceType device,
          std::vector<std::string> labels);
  ~NanoDet();

  torch::Tensor preProcess(cv::Mat *image);
  torch::jit::script::Module network() const;
  torch::Tensor meanTensor() const;
  torch::Tensor stdTensor() const;
  std::vector<std::string> labels() const;
  std::vector<OpenDRDetectionTarget> outputs;
};

NanoDet::NanoDet(torch::jit::script::Module network, torch::Tensor meanValues, torch::Tensor stdValues,
                 torch::DeviceType device, const std::vector<std::string> labels) {
  this->mDevice = device;
  this->mNetwork = network;
  this->mMeanTensor = meanValues.clone().to(device);
  this->mStdTensor = stdValues.clone().to(device);
  this->mLabels = labels;
}

NanoDet::~NanoDet() {
}

/**
 * Helper function for preprocessing images for normalization.
 * This function follows the OpenDR's Nanodet pre-processing pipeline for color normalization.
 * Mean and Standard deviation are already part of NanoDet class when it is initialized.
 * @param image, image to be preprocessed
 */
torch::Tensor NanoDet::preProcess(cv::Mat *image) {
  torch::Tensor tensorImage = torch::from_blob(image->data, {image->rows, image->cols, 3}, torch::kByte);
  tensorImage = tensorImage.toType(torch::kFloat);
  tensorImage = tensorImage.to(this->mDevice);
  tensorImage = tensorImage.permute({2, 0, 1});
  tensorImage = tensorImage.add(this->mMeanTensor);
  tensorImage = tensorImage.mul(this->mStdTensor);

  // divisible padding
  int pad_width = (int((image->cols + 32 - 1) / 32) * 32) - image->cols;
  int pad_height = (int((image->rows + 32 - 1) / 32) * 32) - image->rows;
  torch::nn::functional::PadFuncOptions padding({0, pad_width, 0, pad_height});  // left, right, top, bottom,
  tensorImage = torch::nn::functional::pad(tensorImage, padding);
  tensorImage.unsqueeze_(0);
  return tensorImage;
}

/**
 * Getter for jit model
 */
torch::jit::script::Module NanoDet::network() const {
  return this->mNetwork;
}

/**
 * Getter for tensor with the mean values
 */
torch::Tensor NanoDet::meanTensor() const {
  return this->mMeanTensor;
}

/**
 * Getter for tensor with the standard deviation values
 */
torch::Tensor NanoDet::stdTensor() const {
  return this->mStdTensor;
}

/**
 * Getter of labels
 */
std::vector<std::string> NanoDet::labels() const {
  return this->mLabels;
}

/**
 * Helper function to extract arrays or vectors of integers from JSON files
 * @param json a string of JSON file
 * @param key the key of value to extract from the JSON file
 * @return a vector of integers from extracted key values
 */
std::vector<int> gestIntVectorFromJson(const char *json, const char *key) {
  rapidjson::Document doc;
  doc.Parse(json);

  std::vector<int> items;
  if (!doc.HasMember(key)) {
    if (doc.HasMember("inference_params")) {
      const rapidjson::Value &inferenceParams = doc["inference_params"];
      if (inferenceParams.HasMember(key) && inferenceParams[key].IsArray()) {
        for (rapidjson::SizeType i = 0; i < inferenceParams[key].Size(); i++) {
          if (inferenceParams[key][i].IsInt())
            items.push_back(inferenceParams[key][i].GetInt());
        }
        return items;
      }
    }
    std::cout << key << " is not a member of json or inference_params" << std::endl;
  }
  if (doc[key].IsArray()) {
    for (rapidjson::SizeType i = 0; i < doc[key].Size(); i++) {
      if (doc[key][i].IsInt())
        items.push_back(doc[key][i].GetInt());
    }
    return items;
  }
  std::cout << key << " is not a member of json or it is not an array" << std::endl;
  return items;
}

/**
 * Helper function to extract arrays or vectors of strings from JSON files
 * @param json a string of JSON file
 * @param key the key of value to extract from the JSON file
 * @return a vector of integers from extracted key values
 */
std::vector<std::string> getStringVectorFromJson(const char *json, const char *key) {
  rapidjson::Document doc;
  doc.Parse(json);

  std::vector<std::string> items;
  if (!doc.HasMember(key)) {
    if (doc.HasMember("inference_params")) {
      const rapidjson::Value &inferenceParams = doc["inference_params"];
      if (inferenceParams.HasMember(key) && inferenceParams[key].IsArray()) {
        for (rapidjson::SizeType i = 0; i < inferenceParams[key].Size(); i++) {
          if (inferenceParams[key][i].IsString())
            items.push_back(inferenceParams[key][i].GetString());
        }
        return items;
      }
    }
    std::cout << key << " is not a member of json or inference_params" << std::endl;
  }
  if (doc[key].IsArray()) {
    for (rapidjson::SizeType i = 0; i < doc[key].Size(); i++) {
      if (doc[key][i].IsString())
        items.push_back(doc[key][i].GetString());
    }
    return items;
  }
  std::cout << key << " is not a member of json or it is not an array" << std::endl;
  return items;
}

/**
 * Helper function to calculate the final shape of the model input relative to size ratio of input image.
 */
void getMinimumDstShape(cv::Size *srcSize, cv::Size *dstSize, float divisible) {
  float ratio;
  float srcRatio = ((float)srcSize->width / (float)srcSize->height);
  float dstRatio = ((float)dstSize->width / (float)dstSize->height);
  if (srcRatio < dstRatio)
    ratio = ((float)dstSize->height / (float)srcSize->height);
  else
    ratio = ((float)dstSize->width / (float)srcSize->width);

  dstSize->width = static_cast<int>(ratio * srcSize->width);
  dstSize->height = static_cast<int>(ratio * srcSize->height);

  if (divisible > 0) {
    dstSize->width = std::max(divisible, ((int)((dstSize->width + divisible - 1) / divisible) * divisible));
    dstSize->height = std::max(divisible, ((int)((dstSize->height + divisible - 1) / divisible) * divisible));
  }
}

/**
 * Helper function to calculate the warp matrix for resizing.
 */
void getResizeMatrix(cv::Size *srcShape, cv::Size *dstShape, cv::Mat *Rs, int keepRatio) {
  if (keepRatio == 0) {
    float ratio;
    cv::Mat C = cv::Mat::eye(3, 3, CV_32FC1);

    C.at<float>(0, 2) = -srcShape->width / 2.0;
    C.at<float>(1, 2) = -srcShape->height / 2.0;
    float srcRatio = ((float)srcShape->width / (float)srcShape->height);
    float dstRatio = ((float)dstShape->width / (float)dstShape->height);
    if (srcRatio < dstRatio) {
      ratio = ((float)dstShape->height / (float)srcShape->height);
    } else {
      ratio = ((float)dstShape->width / (float)srcShape->width);
    }

    Rs->at<float>(0, 0) *= ratio;
    Rs->at<float>(1, 1) *= ratio;

    cv::Mat T = cv::Mat::eye(3, 3, CV_32FC1);
    T.at<float>(0, 2) = 0.5 * dstShape->width;
    T.at<float>(1, 2) = 0.5 * dstShape->height;

    *Rs = T * (*Rs) * C;
  } else {
    Rs->at<float>(0, 0) *= (float)dstShape->width / (float)srcShape->width;
    Rs->at<float>(1, 1) *= (float)dstShape->height / (float)srcShape->height;
  }
}

/**
 * Helper function for preprocessing images for resizing.
 * This function follows OpenDR's Nanodet pre-processing pipeline for shape transformation, which includes
 * finding the actual final size of the model input if keep ratio is enabled, calculating the warp matrix and finally
 * resizing and warping the perspective of the input image.
 * @param src, image to be preprocessed
 * @param dst, output image to be used as model input
 * @param dstSize, final size of the dst
 * @param warpMatrix, matrix to be used for warp perspective
 * @param keepRatio, flag for targeting the resized image size relative to input image ratio
 */
void preprocess(cv::Mat *src, cv::Mat *dst, cv::Size *dstSize, cv::Mat *warpMatrix, int keepRatio) {
  cv::Size srcSize = cv::Size(src->cols, src->rows);
  const float divisible = 0.0;

  // Get new destination size if keep ratio is enabled
  if (keepRatio == 0) {
    getMinimumDstShape(&srcSize, dstSize, divisible);
  }

  getResizeMatrix(&srcSize, dstSize, warpMatrix, keepRatio);
  cv::warpPerspective(*src, *dst, *warpMatrix, *dstSize);
}

/**
 * Helper function to determine the device of jit model and tensors.
 */
torch::DeviceType torchDevice(const char *deviceName, int verbose = 0) {
  torch::DeviceType device;
  if (std::string(deviceName) == "cuda") {
    if (verbose == 1)
      printf("to cuda\n");
    device = torch::kCUDA;
  } else {
    if (verbose == 1)
      printf("to cpu\n");
    device = torch::kCPU;
  }
  return device;
}

void loadNanodetModel(const char *modelPath, const char *modelName, const char *device, float scoreThreshold, int height,
                      int width, int keepRatio, NanodetModelT *model) {
  // Initialize model
  model->network = NULL;
  model->scoreThreshold = scoreThreshold;
  model->keepRatio = keepRatio;

  // Parse the model JSON file
  std::string basePath(modelPath);
  std::string modelNameString(modelName);
  std::string modelJsonPath = basePath + "/nanodet_" + modelNameString + ".json";
  std::ifstream inStream(modelJsonPath);
  if (!inStream.is_open()) {
    std::cerr << "Cannot open JSON model file." << std::endl;
    return;
  }
  std::string str((std::istreambuf_iterator<char>(inStream)), std::istreambuf_iterator<char>());
  const char *json = str.c_str();

  // Parse JSON
  std::string jitModelName = jsonGetStringFromKey(json, "model_paths", 0);
  std::string jitModelPath = basePath + "/" + jitModelName;
  std::string modelFormat = jsonGetStringFromKey(json, "format", 0);
  int modelOptimized = jsonGetBoolFromKey(json, "optimized", 0);

  // Proceed only if the model is in onnx format
  if (modelFormat != "pth" || modelOptimized != 0) {
    std::cerr << "Model not in JIT format." << std::endl;
    return;
  }

  // Parse inference params
  const std::vector<int> jsonSize = gestIntVectorFromJson(json, "input_size");
  const std::vector<std::string> labels = getStringVectorFromJson(json, "classes");

  int **colorList = new int *[labels.size()];
  for (int i = 0; i < labels.size(); i++) {
    colorList[i] = new int[3];
  }
  // seed the random number generator
  std::srand(1);
  for (int i = 0; i < labels.size(); i++) {
    for (int j = 0; j < 3; j++) {
      colorList[i][j] = std::rand() % 256;
    }
  }

  // mean and standard deviation tensors for normalization of input
  torch::Tensor meanTensor = torch::tensor({{{-103.53f}}, {{-116.28f}}, {{-123.675f}}});
  torch::Tensor stdValues = torch::tensor({{{0.017429f}}, {{0.017507f}}, {{0.017125f}}});

  // initialization of jit model and class as holder of c++ values.
  torch::DeviceType initDevice = torchDevice(device, 0);
  torch::jit::script::Module network = torch::jit::load(jitModelPath.c_str(), initDevice);
  network.eval();

  NanoDet *detector = new NanoDet(network, meanTensor, stdValues, initDevice, labels);

  model->network = static_cast<void *>(detector);
  model->colorList = colorList;
  model->numberOfClasses = labels.size();

  model->inputSizes[0] = jsonSize[0];
  model->inputSizes[1] = jsonSize[1];

  if (width != 0)
    model->inputSizes[0] = width;
  if (height != 0)
    model->inputSizes[1] = height;
}

void ffNanodet(NanoDet *model, torch::Tensor *inputTensor, cv::Mat *warpMatrix, cv::Size *originalSize,
               torch::Tensor *outputs) {
  // Make all the inputs as tensors to use in jit model
  torch::Tensor srcHeight = torch::tensor(originalSize->height);
  torch::Tensor srcWidth = torch::tensor(originalSize->width);
  torch::Tensor warpMat = torch::from_blob(warpMatrix->data, {3, 3});

  // Model inference
  *outputs = (model->network()).forward({*inputTensor, srcHeight, srcWidth, warpMat}).toTensor();
}

OpenDRDetectionVectorTargetT inferNanodet(NanodetModelT *model, OpenDRImageT *image) {
  //
  NanoDet *networkPTR = static_cast<NanoDet *>(model->network);
  OpenDRDetectionVectorTargetT detectionsVector;
  initDetectionsVector(&detectionsVector);

  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
  if (!opencvImage) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return detectionsVector;
  }

  // Preprocess image and keep values as input in jit model
  cv::Mat resizedImg;
  cv::Size dstSize = cv::Size(model->inputSizes[0], model->inputSizes[1]);
  cv::Mat warpMatrix = cv::Mat::eye(3, 3, CV_32FC1);

  preprocess(opencvImage, &resizedImg, &dstSize, &warpMatrix, model->keepRatio);
  torch::Tensor input = networkPTR->preProcess(&resizedImg);
  cv::Size originalSize(opencvImage->cols, opencvImage->rows);

  torch::Tensor outputs;

  ffNanodet(networkPTR, &input, &warpMatrix, &originalSize, &outputs);

  std::vector<OpenDRDetectionTarget> detections;

  if (outputs.numel() == 0)
    return detectionsVector;

  for (int box = 0; box < outputs.size(0); box++) {
    OpenDRDetectionTargetT detection;
    detection.name = outputs[box][5].item<int>();
    detection.left = outputs[box][0].item<float>();
    detection.top = outputs[box][1].item<float>();
    detection.width = outputs[box][2].item<float>() - outputs[box][0].item<float>();
    detection.height = outputs[box][3].item<float>() - outputs[box][1].item<float>();
    detection.score = outputs[box][4].item<float>();
    detections.push_back(detection);
  }
  // Put vector detection as C pointer and size
  if (static_cast<int>(detections.size()) > 0)
    loadDetectionsVector(&detectionsVector, detections.data(), static_cast<int>(detections.size()));

  return detectionsVector;
}

void drawBboxes(OpenDRImageT *image, NanodetModelT *model, OpenDRDetectionVectorTargetT *vector, int show) {
  int **colorList = model->colorList;

  std::vector<std::string> classNames = (static_cast<NanoDet *>(model->network))->labels();

  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
  if (!opencvImage) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return;
  }

  for (size_t i = 0; i < vector->size; i++) {
    const OpenDRDetectionTarget bbox = (vector->startingPointer)[i];
    float score = bbox.score > 1 ? 1 : bbox.score;
    if (score > model->scoreThreshold) {
      cv::Scalar color = cv::Scalar(colorList[bbox.name][0], colorList[bbox.name][1], colorList[bbox.name][2]);
      cv::rectangle(*opencvImage,
                    cv::Rect(cv::Point(bbox.left, bbox.top), cv::Point((bbox.left + bbox.width), (bbox.top + bbox.height))),
                    color);

      char text[256];

      sprintf(text, "%s %.1f%%", (classNames)[bbox.name].c_str(), score * 100);

      int baseLine = 0;
      cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

      int x = (int)bbox.left;
      int y = (int)bbox.top;
      if (y < 0)
        y = 0;
      if (x + labelSize.width > opencvImage->cols)
        x = opencvImage->cols - labelSize.width;

      cv::rectangle(*opencvImage, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), color, -1);
      cv::putText(*opencvImage, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(255, 255, 255));
    }
  }

  if (show == 0) {
    cv::imshow("image", *opencvImage);
    cv::waitKey(0);
  }
}

void freeNanodetModel(NanodetModelT *model) {
  if (model->network) {
    NanoDet *networkPTR = static_cast<NanoDet *>(model->network);
    delete networkPTR;
    model->network = NULL;
  }

  for (int i = 0; i < model->numberOfClasses; i++) {
    delete[] model->colorList[i];
  }
  delete[] model->colorList;
}

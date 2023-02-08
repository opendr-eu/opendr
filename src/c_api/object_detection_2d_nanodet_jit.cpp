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

#include "object_detection_2d_nanodet_jit.h"

#include <chrono>
#include <document.h>
#include <torch/script.h>
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

void loadNanodetModel(const char *modelPath, const char *modelName, const char *device, float scoreThreshold, int height, int width,
                      NanodetModelT *model) {
  // Initialize model
  model->scoreThreshold = scoreThreshold;
  model->keepRatio = 0;

  // Parse the model JSON file
  std::string basePath(modelPath);
  std::string modelJsonPath = basePath + "/nanodet_" + *modelName + ".json";
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
               std::vector<torch::Tensor> *outputs) {
  // Make all the inputs as tensors to use in jit model
  torch::Tensor srcHeight = torch::tensor(originalSize->height);
  torch::Tensor srcWidth = torch::tensor(originalSize->width);
  torch::Tensor warpMat = torch::from_blob(warpMatrix->data, {3, 3});

  // Model inference
  *outputs = (model->network()).forward({*inputTensor, srcWidth, srcHeight, warpMat}).toTensorVector();
//  *outputs = outputs->to(torch::Device(torch::kCPU, 0));
}

OpenDRDetectionVectorTargetT inferNanodet(NanodetModelT *model, OpenDRImageT *image, double *outFps) {

  auto start = std::chrono::steady_clock::now();
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

  std::vector<torch::Tensor> outputs;

  ffNanodet(networkPTR, &input, &warpMatrix, &originalSize, &outputs);

  std::vector<OpenDRDetectionTarget> detections;
  // Postprocessing, find which outputs have better score than threshold and keep them.


  for (int label = 0; label < outputs.size(); label++) {
    for (int box = 0; box < outputs[label].size(0); box++) {
//      if (outputs[label][box][4].item<float>() > model->scoreThreshold) {
        OpenDRDetectionTargetT detection;
        detection.name = outputs[label][box][5].item<int>();
        detection.left = outputs[label][box][0].item<float>();
        detection.top = outputs[label][box][1].item<float>();
        detection.width = outputs[label][box][2].item<float>() - outputs[label][box][0].item<float>();
        detection.height = outputs[label][box][3].item<float>() - outputs[label][box][1].item<float>();
        detection.score = outputs[label][box][4].item<float>();
        detections.push_back(detection);
//      }
    }
  }
  auto end = std::chrono::steady_clock::now();
  // Put vector detection as C pointer and size
  if (static_cast<int>(detections.size()) > 0)
    loadDetectionsVector(&detectionsVector, detections.data(), static_cast<int>(detections.size()));


  *outFps = 1000000000.0 / ((double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));

  return detectionsVector;
}

void benchmarkNanodet(NanodetModelT *model, OpenDRImageT *image, int repetitions, int warmup) {
  NanoDet *networkPTR = static_cast<NanoDet *>(model->network);
  OpenDRDetectionVectorTargetT detectionsVector;
  initDetectionsVector(&detectionsVector);

  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);

  // Preprocess image and keep values as input in jit model
  cv::Mat resizedImg;
  cv::Size dstSize = cv::Size(model->inputSizes[0], model->inputSizes[1]);
  cv::Mat warpMatrix = cv::Mat::eye(3, 3, CV_32FC1);

  torch::Tensor input;
  preprocess(opencvImage, &resizedImg, &dstSize, &warpMatrix, model->keepRatio);
  input = networkPTR->preProcess(&resizedImg);

  cv::Mat frame(model->inputSizes[1],model->inputSizes[0],CV_8UC3);
  for(int i = 0; i < frame.rows; i++) {
    for(int j = 0; j < frame.cols; j++) {
      frame.at<cv::Vec3b>(i, j)[0] = rand() % 256;
      frame.at<cv::Vec3b>(i, j)[1] = rand() % 256;
      frame.at<cv::Vec3b>(i, j)[2] = rand() % 256;
    }
  }

  OpenDRImageT opImage;
  // Add frame data to OpenDR Image
  if (frame.empty()) {
    opImage.data = NULL;
  } else {
    cv::Mat *tempMatPtr = new cv::Mat(frame);
    opImage.data = (void *)tempMatPtr;
  }

  cv::Mat *tempOpencvImage = static_cast<cv::Mat *>(image->data);
  cv::Mat tempResizedImg;
  cv::Size tempDstSize = cv::Size(model->inputSizes[0], model->inputSizes[1]);
  cv::Mat tempWarpMatrix = cv::Mat::eye(3, 3, CV_32FC1);
  torch::Tensor tempInput;
  double preTimings[repetitions];
    for (int i = 0; i < warmup; i++) {
//    std::cout<<"before warmup preprocess\n";
    preprocess(tempOpencvImage, &tempResizedImg, &tempDstSize, &tempWarpMatrix, model->keepRatio);
    tempInput = networkPTR->preProcess(&tempResizedImg);
  }
  for (int i = 0; i < repetitions; i++) {
    auto start = std::chrono::steady_clock::now();
    preprocess(tempOpencvImage, &tempResizedImg, &tempDstSize, &tempWarpMatrix, model->keepRatio);
    tempInput = networkPTR->preProcess(&tempResizedImg);
    auto end = std::chrono::steady_clock::now();
    preTimings[i] = ((double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
  }

  cv::Size originalSize(opencvImage->cols, opencvImage->rows);

  double inferPostTimings[repetitions];
  std::vector<torch::Tensor> outputs;
  for (int i = 0; i < warmup; i++) {
    ffNanodet(networkPTR, &input, &warpMatrix, &originalSize, &outputs);
  }

  for (int i = 0; i < repetitions; i++) {
    auto start = std::chrono::steady_clock::now();
    ffNanodet(networkPTR, &input, &warpMatrix, &originalSize, &outputs);
    auto end = std::chrono::steady_clock::now();
    inferPostTimings[i] = ((double)(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()));
  }
  // Measure mean time
  double meanInferPostTiming = 0.0;
  double meanPreTiming = 0.0;
  for (int i = 0; i < repetitions; i++) {
    meanInferPostTiming += inferPostTimings[i];
    meanPreTiming += preTimings[i];
  }

  meanInferPostTiming /= repetitions;
  meanPreTiming /= repetitions;

  std::cout<<"C\n\n"
             "=== JIT measurements === \n"
             "preprocessing  fps = "<< (1000000000.0/meanPreTiming) <<" evn/s\n"
             "infer + postpr fps = "<< (1000000000.0/meanInferPostTiming) <<" evn/s\n\n";


}

void drawBboxes(OpenDRImageT *image, NanodetModelT *model, OpenDRDetectionVectorTargetT *vector) {
  int **colorList = model->colorList;

  std::vector<std::string> classNames = (static_cast<NanoDet *>(model->network))->labels();

  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
  if (!opencvImage) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return;
  }

  cv::Mat imageWithDetections = (*opencvImage).clone();
  for (size_t i = 0; i < vector->size; i++) {
    const OpenDRDetectionTarget bbox = (vector->startingPointer)[i];
    float score = bbox.score > 1 ? 1 : bbox.score;
    if (score > model->scoreThreshold) {
      cv::Scalar color = cv::Scalar(colorList[bbox.name][0], colorList[bbox.name][1], colorList[bbox.name][2]);
      cv::rectangle(imageWithDetections,
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
      if (x + labelSize.width > imageWithDetections.cols)
        x = imageWithDetections.cols - labelSize.width;

      cv::rectangle(imageWithDetections, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                    color, -1);
      cv::putText(imageWithDetections, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(255, 255, 255));
    }
  }

  cv::imshow("image", imageWithDetections);
  cv::waitKey(0);
}

void drawBboxesWithFps(OpenDRImageT *image, NanodetModelT *model, OpenDRDetectionVectorTargetT *vector, double fps) {
  int **colorList = model->colorList;

  std::vector<std::string> classNames = (static_cast<NanoDet *>(model->network))->labels();

  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
  if (!opencvImage) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return;
  }

  cv::Mat imageWithDetections = (*opencvImage).clone();
  for (size_t i = 0; i < vector->size; i++) {
    const OpenDRDetectionTarget bbox = (vector->startingPointer)[i];
    float score = bbox.score > 1 ? 1 : bbox.score;
    if (score > model->scoreThreshold) {
      cv::Scalar color = cv::Scalar(colorList[bbox.name][0], colorList[bbox.name][1], colorList[bbox.name][2]);
      cv::rectangle(imageWithDetections,
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
      if (x + labelSize.width > imageWithDetections.cols)
        x = imageWithDetections.cols - labelSize.width;

      cv::rectangle(imageWithDetections, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
                    color, -1);
      cv::putText(imageWithDetections, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.4,
                  cv::Scalar(255, 255, 255));

      // Put fps counter
      std::string fpsText = "FPS: " + std::to_string(fps);
      cv::putText(imageWithDetections, fpsText, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1,
                  cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
    }
  }

  cv::imshow("image", imageWithDetections);
  cv::waitKey(1);
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

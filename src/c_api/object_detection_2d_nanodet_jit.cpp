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
  std::vector<OpendrDetectionTarget> outputs;
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
  if (keepRatio == 1) {
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
  if (keepRatio == 1) {
    getMinimumDstShape(&srcSize, dstSize, divisible);
  }

  getResizeMatrix(&srcSize, dstSize, warpMatrix, keepRatio);
  cv::warpPerspective(*src, *dst, *warpMatrix, *dstSize);
}

/**
 * Helper function to determine the device of jit model and tensors.
 */
torch::DeviceType torchDevice(char *deviceName, int verbose = 0) {
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

void loadNanodetModel(char *modelPath, char *device, int height, int width, float scoreThreshold, NanodetModelT *model) {
  // Initialize model
  model->inputSizes[0] = width;
  model->inputSizes[1] = height;

  model->scoreThreshold = scoreThreshold;
  model->keepRatio = 1;

  const std::vector<std::string> labels{
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

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
  torch::jit::script::Module network = torch::jit::load(modelPath, initDevice);
  network.eval();

  NanoDet *detector = new NanoDet(network, meanTensor, stdValues, initDevice, labels);

  model->network = static_cast<void *>(detector);
  model->colorList = colorList;
  model->numberOfClasses = labels.size();
}

void ffNanodet(NanoDet *model, torch::Tensor *inputTensor, cv::Mat *warpMatrix, cv::Size *originalSize,
               torch::Tensor *outputs) {
  // Make all the inputs as tensors to use in jit model
  torch::Tensor srcHeight = torch::tensor(originalSize->height);
  torch::Tensor srcWidth = torch::tensor(originalSize->width);
  torch::Tensor warpMat = torch::from_blob(warpMatrix->data, {3, 3});

  // Model inference
  *outputs = (model->network()).forward({*inputTensor, srcWidth, srcHeight, warpMat}).toTensor();
  *outputs = outputs->to(torch::Device(torch::kCPU, 0));
}

OpendrDetectionVectorTargetT inferNanodet(NanodetModelT *model, OpendrImageT *image) {
  NanoDet *networkPTR = static_cast<NanoDet *>(model->network);
  OpendrDetectionVectorTargetT detectionsVector;
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

  std::vector<OpendrDetectionTarget> detections;
  // Postprocessing, find which outputs have better score than threshold and keep them.
  for (int label = 0; label < outputs.size(0); label++) {
    for (int box = 0; box < outputs.size(1); box++) {
      if (outputs[label][box][4].item<float>() > model->scoreThreshold) {
        OpendrDetectionTargetT detection;
        detection.name = label;
        detection.left = outputs[label][box][0].item<float>();
        detection.top = outputs[label][box][1].item<float>();
        detection.width = outputs[label][box][2].item<float>() - outputs[label][box][0].item<float>();
        detection.height = outputs[label][box][3].item<float>() - outputs[label][box][1].item<float>();
        detection.score = outputs[label][box][4].item<float>();
        detections.push_back(detection);
      }
    }
  }

  // Put vector detection as C pointer and size
  if (static_cast<int>(detections.size()) > 0)
    loadDetectionsVector(&detectionsVector, detections.data(), static_cast<int>(detections.size()));

  return detectionsVector;
}

void drawBboxes(OpendrImageT *image, NanodetModelT *model, OpendrDetectionVectorTargetT *vector) {
  int **colorList = model->colorList;

  std::vector<std::string> classNames = (static_cast<NanoDet *>(model->network))->labels();

  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
  if (!opencvImage) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return;
  }

  cv::Mat imageWithDetections = (*opencvImage).clone();
  for (size_t i = 0; i < vector->size; i++) {
    const OpendrDetectionTarget bbox = (vector->startingPointer)[i];
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

void freeNanodetModel(NanodetModelT *model) {
  if (model->network) {
    NanoDet *networkPTR = static_cast<NanoDet *>(model->network);
    delete networkPTR;
  }

  for (int i = 0; i < model->numberOfClasses; i++) {
    delete[] model->colorList[i];
  }
  delete[] model->colorList;
}

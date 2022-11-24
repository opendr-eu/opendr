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

#include <torch/script.h>
#include <torchvision/vision.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "nanodet_c.h"

/**
 * Helper class holder of c++ values and jit model.
 */
class NanoDet {
private:
  torch::DeviceType device;
  torch::jit::script::Module network;
  torch::Tensor meanTensor;
  torch::Tensor stdTensor;
  std::vector<std::string> labels;

public:
  NanoDet(torch::jit::script::Module net, torch::Tensor meanValues, torch::Tensor stdValues, torch::DeviceType device,
          std::vector<std::string> labels);
  ~NanoDet();

  torch::Tensor mPreProcess(cv::Mat *image);
  torch::jit::script::Module net() const;
  torch::Tensor meanValues() const;
  torch::Tensor stdValues() const;
  std::vector<std::string> classes() const;
  std::vector<opendr_detection_target> outputs;
};

NanoDet::NanoDet(torch::jit::script::Module net, torch::Tensor meanValues, torch::Tensor stdValues, torch::DeviceType device,
                 const std::vector<std::string> labels) {
  this->device = device;
  this->network = net;
  this->meanTensor = meanValues.clone().to(device);
  this->stdTensor = stdValues.clone().to(device);
  this->labels = labels;
}

NanoDet::~NanoDet() {
}

/**
 * Helper function for preprocessing images for normalization.
 * This function follows the OpenDR's Nanodet pre-processing pipeline for color normalization.
 * Mean and Standard deviation are already part of NanoDet class when is initialized.
 * @param image, image to be preprocesses
 */
torch::Tensor NanoDet::mPreProcess(cv::Mat *image) {
  torch::Tensor tensorImage = torch::from_blob(image->data, {image->rows, image->cols, 3}, torch::kByte);
  tensorImage = tensorImage.toType(torch::kFloat);
  tensorImage = tensorImage.to(this->device);
  tensorImage = tensorImage.permute({2, 0, 1});
  tensorImage = tensorImage.add(this->meanTensor);
  tensorImage = tensorImage.mul(this->stdTensor);

  return tensorImage;
}

/**
 * Getter for jit model
 */
torch::jit::script::Module NanoDet::net() const {
  return this->network;
}

/**
 * Getter for tensor with the mean values
 */
torch::Tensor NanoDet::meanValues() const {
  return this->meanTensor;
}

/**
 * Getter for tensor with the standard deviation values
 */
torch::Tensor NanoDet::stdValues() const {
  return this->stdTensor;
}

/**
 * Getter of labels for printing
 */
std::vector<std::string> NanoDet::classes() const {
  return labels;
}

/**
 * Helper function to calculate the final shape of the model input relative to size ratio of input image.
 */
void get_minimum_dst_shape(cv::Size *srcSize, cv::Size *dstSize, float divisible) {
  float ratio;
  float src_ratio = ((float)srcSize->width / (float)srcSize->height);
  float dst_ratio = ((float)dstSize->width / (float)dstSize->height);
  if (src_ratio < dst_ratio)
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
void get_resize_matrix(cv::Size *src_shape, cv::Size *dst_shape, cv::Mat *Rs, int keep_ratio) {
  if (keep_ratio == 1) {
    float ratio;
    cv::Mat C = cv::Mat::eye(3, 3, CV_32FC1);

    C.at<float>(0, 2) = -src_shape->width / 2.0;
    C.at<float>(1, 2) = -src_shape->height / 2.0;
    float src_ratio = ((float)src_shape->width / (float)src_shape->height);
    float dst_ratio = ((float)dst_shape->width / (float)dst_shape->height);
    if (src_ratio < dst_ratio) {
      ratio = ((float)dst_shape->height / (float)src_shape->height);
    } else {
      ratio = ((float)dst_shape->width / (float)src_shape->width);
    }

    Rs->at<float>(0, 0) *= ratio;
    Rs->at<float>(1, 1) *= ratio;

    cv::Mat T = cv::Mat::eye(3, 3, CV_32FC1);
    T.at<float>(0, 2) = 0.5 * dst_shape->width;
    T.at<float>(1, 2) = 0.5 * dst_shape->height;

    *Rs = T * (*Rs) * C;
  } else {
    Rs->at<float>(0, 0) *= (float)dst_shape->width / (float)src_shape->width;
    Rs->at<float>(1, 1) *= (float)dst_shape->height / (float)src_shape->height;
  }
}

/**
 * Helper function for preprocessing images for resizing.
 * This function follows the OpenDR's Nanodet pre-processing pipeline for shape transformation, which include
 * find the actual final size of model input if keep ratio is enabled, calculate the warp matrix and finally
 * resize and warp perspective of the input image.
 * @param src, image to be preprocesses
 * @param dst, output image to be used as model input
 * @param dstSize, final size of the dst
 * @param Rs, matrix to be used for warp perspective
 * @param keep_ratio, flag for targeting the resized image size relative to input image ratio
 */
void preprocess(cv::Mat *src, cv::Mat *dst, cv::Size *dstSize, cv::Mat *warp_matrix, int keep_ratio) {
  cv::Size srcSize = cv::Size(src->cols, src->rows);
  const float divisible = 0.0;

  // Get new destination size if keep ratio is wanted
  if (keep_ratio == 1) {
    get_minimum_dst_shape(&srcSize, dstSize, divisible);
  }

  get_resize_matrix(&srcSize, dstSize, warp_matrix, keep_ratio);
  cv::warpPerspective(*src, *dst, *warp_matrix, *dstSize);
}

/**
 * Helper function to determine the device of jit model and tensors.
 */
torch::DeviceType torchDevice(char *device_name, int verbose = 0) {
  torch::DeviceType device;
  if (std::string(device_name) == "cuda") {
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

void load_nanodet_model(char *model_path, char *device, int height, int width, float scoreThreshold, nanodet_model_t *model) {
  // Initialize model
  model->inputSize[0] = width;
  model->inputSize[1] = height;

  model->scoreThreshold = scoreThreshold;
  model->keep_ratio = 1;

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

  // mean and standard deviation tensors for normalization of input
  torch::Tensor meanTensor = torch::tensor({{{-103.53f}}, {{-116.28f}}, {{-123.675f}}});
  torch::Tensor stdValues = torch::tensor({{{0.017429f}}, {{0.017507f}}, {{0.017125f}}});

  // initialization of jit model and class as holder of c++ values.
  torch::DeviceType torch_device = torchDevice(device, 1);
  torch::jit::script::Module net = torch::jit::load(model_path, torch_device);
  net.eval();

  NanoDet *detector = new NanoDet(net, meanTensor, stdValues, torch_device, labels);

  model->net = static_cast<void *>detector;
}

opendr_detection_target_list_t infer_nanodet(opendr_image_t *image, nanodet_model_t *model) {
  NanoDet *networkPTR = static_cast<NanoDet *>(model->net);
  opendr_detection_target_list_t detections;

  std::vector<opendr_detection_target> dets;
  cv::Mat *opencv_image = static_cast<cv::Mat *>(image->data);
  if (!opencv_image) {
    std::cerr << "Cannot load image for inference." << std::endl;

    // Initialize an empty detection to return.
    initialize_detections(&detections);
    return detections;
  }

  // Preprocess image and keep values as input in jit model
  cv::Mat resizedImg;
  cv::Size dstSize = cv::Size(model->inputSize[0], model->inputSize[1]);
  cv::Mat warp_matrix = cv::Mat::eye(3, 3, CV_32FC1);
  preprocess(opencv_image, &resizedImg, &dstSize, &warp_matrix, model->keep_ratio);
  torch::Tensor input = networkPTR->mPreProcess(&resizedImg);

  // Make all the inputs as tensors to use in jit model
  torch::Tensor srcHeight = torch::tensor(opencv_image->rows);
  torch::Tensor srcWidth = torch::tensor(opencv_image->cols);
  torch::Tensor warpMatrix = torch::from_blob(warp_matrix.data, {3, 3});

  // Model inference
  torch::Tensor outputs = (networkPTR->net()).forward({input, srcHeight, srcWidth, warpMatrix}).toTensor();
  outputs = outputs.to(torch::Device(torch::kCPU, 0));

  // Postprocessing, find which outputs have better score than threshold and keep them.
  for (int label = 0; label < outputs.size(0); label++) {
    for (int box = 0; box < outputs.size(1); box++) {
      if (outputs[label][box][4].item<float>() > model->scoreThreshold) {
        opendr_detection_target_t det;
        det.name = label;
        det.left = outputs[label][box][0].item<float>();
        det.top = outputs[label][box][1].item<float>();
        det.width = outputs[label][box][2].item<float>() - outputs[label][box][0].item<float>();
        det.height = outputs[label][box][3].item<float>() - outputs[label][box][1].item<float>();
        det.score = outputs[label][box][4].item<float>();
        dets.push_back(det);
      }
    }
  }

  // Put vector detection as C pointer and size
  if ((int)dets.size() > 0)
    load_detections(&detections, dets.data(), (int)dets.size());
  else
    initialize_detections(&detections);

  return detections;
}

void drawBboxes(opendr_image_t *opendr_image, nanodet_model_t *model, opendr_detection_target_list_t *detections) {
  const int colorList[80][3] = {
    //{255 ,255 ,255}, //bg
    {216, 82, 24},   {236, 176, 31},  {125, 46, 141},  {118, 171, 47},  {76, 189, 237},  {238, 19, 46},   {76, 76, 76},
    {153, 153, 153}, {255, 0, 0},     {255, 127, 0},   {190, 190, 0},   {0, 255, 0},     {0, 0, 255},     {170, 0, 255},
    {84, 84, 0},     {84, 170, 0},    {84, 255, 0},    {170, 84, 0},    {170, 170, 0},   {170, 255, 0},   {255, 84, 0},
    {255, 170, 0},   {255, 255, 0},   {0, 84, 127},    {0, 170, 127},   {0, 255, 127},   {84, 0, 127},    {84, 84, 127},
    {84, 170, 127},  {84, 255, 127},  {170, 0, 127},   {170, 84, 127},  {170, 170, 127}, {170, 255, 127}, {255, 0, 127},
    {255, 84, 127},  {255, 170, 127}, {255, 255, 127}, {0, 84, 255},    {0, 170, 255},   {0, 255, 255},   {84, 0, 255},
    {84, 84, 255},   {84, 170, 255},  {84, 255, 255},  {170, 0, 255},   {170, 84, 255},  {170, 170, 255}, {170, 255, 255},
    {255, 0, 255},   {255, 84, 255},  {255, 170, 255}, {42, 0, 0},      {84, 0, 0},      {127, 0, 0},     {170, 0, 0},
    {212, 0, 0},     {255, 0, 0},     {0, 42, 0},      {0, 84, 0},      {0, 127, 0},     {0, 170, 0},     {0, 212, 0},
    {0, 255, 0},     {0, 0, 42},      {0, 0, 84},      {0, 0, 127},     {0, 0, 170},     {0, 0, 212},     {0, 0, 255},
    {0, 0, 0},       {36, 36, 36},    {72, 72, 72},    {109, 109, 109}, {145, 145, 145}, {182, 182, 182}, {218, 218, 218},
    {0, 113, 188},   {80, 182, 188},  {127, 127, 0},
  };

  std::vector<std::string> classNames = (static_cast<NanoDet *>(model->net))->classes();

  cv::Mat *opencv_image = static_cast<cv::Mat *>(opendr_image->data);
  if (!opencv_image) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return;
  }

  cv::Mat image = (*opencv_image).clone();
  for (size_t i = 0; i < detections->size; i++) {
    float score = bbox.score > 1 ? 1 : bbox.score;
    if (score > model->scoreThreshold) {
      const opendr_detection_target bbox = (detections->starting_pointer)[i];
      cv::Scalar color = cv::Scalar(colorList[bbox.name][0], colorList[bbox.name][1], colorList[bbox.name][2]);
      cv::rectangle(
        image, cv::Rect(cv::Point(bbox.left, bbox.top), cv::Point((bbox.left + bbox.width), (bbox.top + bbox.height))), color);

      char text[256];

      sprintf(text, "%s %.1f%%", (classNames)[bbox.name].c_str(), score * 100);

      int baseLine = 0;
      cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

      int x = (int)bbox.left;
      int y = (int)bbox.top;
      if (y < 0)
        y = 0;
      if (x + labelSize.width > image.cols)
        x = image.cols - labelSize.width;

      cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), color, -1);
      cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }
  }

  cv::imshow("image", image);
  cv::waitKey(0);
}

void free_nanodet_model(nanodet_model_t *model) {
  if (model->net) {
    NanoDet *networkPTR = static_cast<NanoDet *>(model->net);
    delete networkPTR;
  }
}

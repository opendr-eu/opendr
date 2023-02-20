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

#include "object_detection_2d_yolov5.h"
#include "data.h"
#include "target.h"

#include <document.h>
#include <onnxruntime_cxx_api.h>
#include <codecvt>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <cstring>
#include <iostream>
#include <vector>

size_t vectorProduct(const std::vector<int64_t> &vector) {
  if (vector.empty())
    return 0;

  size_t product = 1;
  for (const auto &element : vector)
    product *= element;

  return product;
}

void letterbox(cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape = cv::Size(640, 640),
               const cv::Scalar &color = cv::Scalar(114, 114, 114), bool auto_ = true, bool scaleFill = false,
               bool scaleUp = true, int stride = 32) {
  cv::Size shape = image.size();
  float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
  if (!scaleUp)
    r = std::min(r, 1.0f);

  int newUnpad[2]{(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

  auto dw = (float)(newShape.width - newUnpad[0]);
  auto dh = (float)(newShape.height - newUnpad[1]);

  if (auto_) {
    dw = (float)((int)dw % stride);
    dh = (float)((int)dh % stride);
  } else if (scaleFill) {
    dw = 0.0f;
    dh = 0.0f;
    newUnpad[0] = newShape.width;
    newUnpad[1] = newShape.height;
  }

  dw /= 2.0f;
  dh /= 2.0f;

  if (shape.width != newUnpad[0] && shape.height != newUnpad[1]) {
    cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
  }

  int top = int(std::round(dh - 0.1f));
  int bottom = int(std::round(dh + 0.1f));
  int left = int(std::round(dw - 0.1f));
  int right = int(std::round(dw + 0.1f));
  cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

template<typename T> T clip(const T &n, const T &lower, const T &upper) {
  return std::max(lower, std::min(n, upper));
}

void scaleCoords(const cv::Size &imageShape, cv::Rect &coords, const cv::Size &imageOriginalShape) {
  float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height,
                        (float)imageShape.width / (float)imageOriginalShape.width);

  int pad[2] = {(int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f),
                (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

  coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
  coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

  coords.width = (int)std::round(((float)coords.width / gain));
  coords.height = (int)std::round(((float)coords.height / gain));

  // // clip coords, should be modified for width and height
  coords.x = clip(coords.x, 0, imageOriginalShape.width);
  coords.y = clip(coords.y, 0, imageOriginalShape.height);
  coords.width = clip((coords.width + coords.x), 0, imageOriginalShape.width) - coords.x;
  coords.height = clip((coords.height + coords.y), 0, imageOriginalShape.height) - coords.y;
}

void getBestClassInfo(std::vector<float>::iterator it, const int &numClasses, float &bestConf, int &bestClassId) {
  // first 5 element are box and obj confidence
  bestClassId = 5;
  bestConf = 0;

  for (int i = 5; i < numClasses + 5; i++) {
    if (it[i] > bestConf) {
      bestConf = it[i];
      bestClassId = i - 5;
    }
  }
}

/**
 * Helper function for preprocessing images before feeding them into the face recognition model.
 * This function follows the OpenDR's face recognition pre-processing pipeline, which includes the following:
 * a) resizing the image into resizeTarget x resizeTarget pixels and then taking a center crop of size modelInputSize,
 * and b) normalizing the resulting values using meanValue and stdValue
 * @param image image to be preprocesses
 * @param blob pre-processed data in a flattened vector
 * @param inputTensorShape actual shape of input Tensor of the Network
 * @param width target width size for resizing
 * @param height target height size for resizing
 * @param isDynamicInputShapeInt integer used as boolean, if zero the model has dynamic input shape
 */
void preprocessYolov5(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape, int width = 680, int height = 680,
                      int isDynamicInputShapeInt = 1) {
  // Convert to RGB
  cv::Mat resizedImage, floatImage;
  cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
  bool isDynamicInputShape = false;
  if (isDynamicInputShapeInt == 0) {
    isDynamicInputShape = true;
  }
  cv::Size2f inputImageShape = cv::Size2f(width, height);
  letterbox(resizedImage, resizedImage, inputImageShape, cv::Scalar(114, 114, 114), isDynamicInputShape, false, true, 32);

  inputTensorShape[2] = resizedImage.rows;
  inputTensorShape[3] = resizedImage.cols;

  // Resize and then get a center crop
  resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
  blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
  cv::Size floatImageSize{floatImage.cols, floatImage.rows};

  // hwc -> chw
  std::vector<cv::Mat> chw(floatImage.channels());
  for (int i = 0; i < floatImage.channels(); ++i) {
    chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
  }
  cv::split(floatImage, chw);
}

void loadYolov5Model(const char *modelPath, const char *modelName, const char *device, float confThreshold, float iouThreshold,
                     int height, int width, Yolov5ModelT *model) {
  // Initialize model
  model->onnxSession = model->env = model->sessionOptions = NULL;
  model->confThreshold = confThreshold;
  model->iouThreshold = iouThreshold;

  // Parse the model JSON file
  std::string basePath(modelPath);
  std::string modelJsonPath = basePath + "/yolov5" + *modelName + ".json";
  std::ifstream inStream(modelJsonPath);
  if (!inStream.is_open()) {
    std::cerr << "Cannot open JSON model file." << std::endl;
    return;
  }
  std::string str((std::istreambuf_iterator<char>(inStream)), std::istreambuf_iterator<char>());
  const char *json = str.c_str();

  // Parse JSON
  std::string onnxModelPath = basePath + "/" + jsonGetStringFromKey(json, "model_paths", 0);
  std::string modelFormat = jsonGetStringFromKey(json, "format", 0);

  // Parse inference thresholds
  if (model->confThreshold <= 0) {
    float threshold = jsonGetFloatFromKeyInInferenceParams(json, "conf_threshold", 0);
    model->confThreshold = threshold;
  }
  if (model->iouThreshold <= 0) {
    float threshold = jsonGetFloatFromKeyInInferenceParams(json, "iou_threshold", 0);
    model->iouThreshold = threshold;
  }

  // Proceed only if the model is in onnx format
  if (modelFormat != "onnx") {
    std::cerr << "Model not in ONNX format." << std::endl;
    return;
  }

  // Find classes and make colors for each label
  OpenDRStringsVector labels;
  initOpenDRStringsVector(&labels);
  getStringVectorFromJson(json, "classes", &labels);

  int **colorList = new int *[labels.size];
  for (int i = 0; i < labels.size; i++) {
    colorList[i] = new int[3];
  }
  // seed the random number generator
  std::srand(1);
  for (int i = 0; i < labels.size; i++) {
    for (int j = 0; j < 3; j++) {
      colorList[i][j] = std::rand() % 256;
    }
  }

  model->labels = labels;
  model->colorList = colorList;
  model->numberOfClasses = labels.size;

  Ort::Env *env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OpenDR_env");

  Ort::SessionOptions *sessionOptions = new Ort::SessionOptions;
  sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  // Find if cuda is available
  std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
  auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
  OrtCUDAProviderOptions cudaOption;

  if ((std::string(device) == "cuda") && (cudaAvailable == availableProviders.end())) {
    std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
    std::cout << "Inference device: CPU" << std::endl;
  } else if ((std::string(device) == "cuda") && (cudaAvailable != availableProviders.end())) {
    std::cout << "Inference device: GPU" << std::endl;
    sessionOptions->AppendExecutionProvider_CUDA(cudaOption);
  } else {
    std::cout << "Inference device: CPU" << std::endl;
  }

  Ort::Session *session = new Ort::Session(*env, onnxModelPath.c_str(), *sessionOptions);
  model->env = env;
  model->onnxSession = session;
  model->sessionOptions = sessionOptions;

  // Export input model size from json file if not else provided
  OpenDRIntsVector jsonSize;
  initOpenDRIntsVector(&jsonSize);
  gestIntVectorFromJson(json, "input_size", &jsonSize);
  model->inputSizes[0] = jsonSize.data[0];
  model->inputSizes[1] = jsonSize.data[1];
  if (width != 0)
    model->inputSizes[0] = width;
  if (height != 0)
    model->inputSizes[1] = height;

  // Find if model has dynamicInput
  model->isDynamicInputShape = 1;
  Ort::TypeInfo inputTypeInfo = session->GetInputTypeInfo(0);
  std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
  // checking if width and height are dynamic
  if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1) {
    std::cout << "Dynamic input shape" << std::endl;
    model->isDynamicInputShape = 0;
  }
}

void postprocessYolov5(const cv::Size &resizedImageShape, const cv::Size &originalImageShape,
                       std::vector<Ort::Value> &outputTensors, const float &confThreshold, const float &iouThreshold,
                       OpenDRDetectionVectorTargetT &detectionsVector) {
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<int> classIds;

  auto *rawOutput = outputTensors[0].GetTensorData<float>();
  std::vector<int64_t> outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
  size_t count = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<float> output(rawOutput, rawOutput + count);

  // first 5 elements are box[4] and obj confidence
  int numClasses = (int)outputShape[2] - 5;
  int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

  // only for batch size = 1
  for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2]) {
    float clsConf = it[4];

    if (clsConf > confThreshold) {
      int centerX = (int)(it[0]);
      int centerY = (int)(it[1]);
      int width = (int)(it[2]);
      int height = (int)(it[3]);
      int left = centerX - width / 2;
      int top = centerY - height / 2;

      float objConf;
      int classId;
      getBestClassInfo(it, numClasses, objConf, classId);

      float confidence = clsConf * objConf;

      boxes.emplace_back(left, top, width, height);
      confidences.emplace_back(confidence);
      classIds.emplace_back(classId);
    }
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, confThreshold, iouThreshold, indices);

  std::vector<OpenDRDetectionTarget> detections;

  for (int idx : indices) {
    cv::Rect opencvBoxes(boxes[idx]);
    scaleCoords(resizedImageShape, opencvBoxes, originalImageShape);

    OpenDRDetectionTarget detection;
    detection.name = classIds[idx];
    detection.left = opencvBoxes.x;
    detection.top = opencvBoxes.y;
    detection.width = opencvBoxes.width;
    detection.height = opencvBoxes.height;
    detection.score = confidences[idx];

    detections.push_back(detection);
  }

  if (static_cast<int>(detections.size()) > 0)
    loadDetectionsVector(&detectionsVector, detections.data(), static_cast<int>(detections.size()));
}

void ffYolov5(Yolov5ModelT *model, std::vector<float> &inputTensorValues, size_t &inputTensorSize,
              std::vector<int64_t> &inputTensorShape, std::vector<Ort::Value> &outputTensors) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);
  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  std::vector<Ort::Value> inputTensors;

  std::vector<const char *> inputNodeNames = {"images"};
  std::vector<const char *> outputNodeNames = {"output"};

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

  inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize,
                                                         inputTensorShape.data(), inputTensorShape.size()));

  outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), inputTensors.data(), 1, outputNodeNames.data(), 1);
}

OpenDRDetectionVectorTargetT inferYolov5(Yolov5ModelT *model, OpenDRImageT *image) {
  float *blob = nullptr;
  std::vector<int64_t> inputTensorShape{1, 3, -1, -1};

  OpenDRDetectionVectorTargetT detectionsVector;
  initDetectionsVector(&detectionsVector);
  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
  if (!opencvImage) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return detectionsVector;
  }

  preprocessYolov5(*opencvImage, blob, inputTensorShape, model->inputSizes[0], model->inputSizes[1],
                   model->isDynamicInputShape);

  size_t inputTensorSize = vectorProduct(inputTensorShape);

  std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
  std::vector<Ort::Value> outputTensors;
  ffYolov5(model, inputTensorValues, inputTensorSize, inputTensorShape, outputTensors);

  cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
  postprocessYolov5(resizedShape, opencvImage->size(), outputTensors, model->confThreshold, model->iouThreshold,
                    detectionsVector);

  delete[] blob;

  return detectionsVector;
}

void freeYolov5Model(Yolov5ModelT *model) {
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

  if (model->labels.data) {
    freeStringsVector(&(model->labels));
  }

  for (int i = 0; i < model->numberOfClasses; i++)
    delete[] model->colorList[i];
  delete[] model->colorList;
}

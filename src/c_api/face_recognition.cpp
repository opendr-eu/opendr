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

#include "face_recognition.h"
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
 * Helper function for preprocessing images before feeding them into the face recognition model.
 * This function follows the OpenDR's face recognition pre-processing pipeline, which includes the following:
 * a) resizing the image into resizeTarget x resizeTarget pixels and then taking a center crop of size modelInputSize,
 * and b) normalizing the resulting values using meanValue and stdValue
 * @param image image to be preprocesses
 * @param data pre-processed data in a flattened vector
 * @param resizeTarget target size for resizing
 * @param modelInputSize size of the center crop (equals the size that the DL model expects)
 * @param meanValue value used for centering the input image
 * @param stdValue value used for scaling the input image
 */
void preprocessFaceRecognition(cv::Mat *image, std::vector<float> &data, int resizeTarget = 128, int modelInputSize = 112,
                               float meanValue = 0.5, float stdValue = 0.5) {
  // Convert to RGB
  cv::Mat normalizedImage;
  cv::cvtColor(*image, normalizedImage, cv::COLOR_BGR2RGB);

  // Resize and then get a center crop
  cv::resize(normalizedImage, normalizedImage, cv::Size(resizeTarget, resizeTarget));
  int stride = (resizeTarget - modelInputSize) / 2;
  cv::Rect myROI(stride, stride, resizeTarget - stride, resizeTarget - stride);
  normalizedImage = normalizedImage(myROI);

  // Scale to 0...1
  cv::Mat outputImage;
  normalizedImage.convertTo(outputImage, CV_32FC3, 1 / 255.0);
  // Unfold the image into the appropriate format
  // This is certainly not the most efficient way to do this...
  // ... and is probably constantly leading to cache misses
  // ... but it works for now.
  for (unsigned int j = 0; j < modelInputSize; ++j) {
    for (unsigned int k = 0; k < modelInputSize; ++k) {
      cv::Vec3f currentPixel = outputImage.at<cv::Vec3f>(j, k);
      data[0 * modelInputSize * modelInputSize + j * modelInputSize + k] = (currentPixel[0] - meanValue) / stdValue;
      data[1 * modelInputSize * modelInputSize + j * modelInputSize + k] = (currentPixel[1] - meanValue) / stdValue;
      data[2 * modelInputSize * modelInputSize + j * modelInputSize + k] = (currentPixel[2] - meanValue) / stdValue;
    }
  }
}

void loadFaceRecognitionModel(const char *modelPath, FaceRecognitionModelT *model) {
  // Initialize model
  model->onnxSession = model->env = model->sessionOptions = NULL;
  model->database = model->databaseIds = NULL;
  model->personNames = NULL;
  model->threshold = 1;

  // Parse the model JSON file
  std::string modelJsonPath(modelPath);
  std::size_t splitPosition = modelJsonPath.find_last_of("/");
  splitPosition = splitPosition > 0 ? splitPosition + 1 : 0;
  modelJsonPath = modelJsonPath + "/" + modelJsonPath.substr(splitPosition) + ".json";

  std::ifstream inStream(modelJsonPath);
  if (!inStream.is_open()) {
    std::cerr << "Cannot open JSON model file" << std::endl;
    return;
  }
  std::string str((std::istreambuf_iterator<char>(inStream)), std::istreambuf_iterator<char>());
  const char *json = str.c_str();

  std::string basePath = modelJsonPath.substr(0, splitPosition);
  splitPosition = basePath.find_last_of("/");
  splitPosition = splitPosition > 0 ? splitPosition + 1 : 0;
  if (splitPosition < basePath.size())
    basePath.resize(splitPosition);

  // Parse JSON
  std::string onnxModelPath = basePath + jsonGetStringFromKey(json, "model_paths", 0);
  std::string modelFormat = jsonGetStringFromKey(json, "format", 0);

  // Parse inference params
  float threshold = jsonGetFloatFromKeyInInferenceParams(json, "threshold", 0);
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
  model->onnxSession = session;
  model->sessionOptions = sessionOptions;

  // Should we pass these parameters through the model json file?
  model->modelSize = 112;
  model->resizeSize = 128;
  model->meanValue = 0.5;
  model->stdValue = 0.5;
  model->outputSize = 128;
}

void freeFaceRecognitionModel(FaceRecognitionModelT *model) {
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

  if (model->databaseIds) {
    delete[] model->databaseIds;
  }

  if (model->database) {
    cv::Mat *database = static_cast<cv::Mat *>(model->database);
    delete database;
  }

  if (model->personNames) {
    for (int i = 0; i < model->nPersons; i++)
      delete[] model->personNames[i];
    delete[] model->personNames;
  }
}

void ffFaceRecognition(FaceRecognitionModelT *model, OpendrImageT *image, cv::Mat *features) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnxSession);
  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  std::vector<int64_t> inputNodeDims = {1, 3, model->modelSize, model->modelSize};
  size_t inputTensorSize = model->modelSize * model->modelSize * 3;

  // Get the input image and pre-process it
  std::vector<float> inputTensorValues(inputTensorSize);
  cv::Mat *opencvImage = static_cast<cv::Mat *>(image->data);
  if (!opencvImage) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return;
  }

  preprocessFaceRecognition(opencvImage, inputTensorValues, model->resizeSize, model->modelSize, model->meanValue,
                            model->stdValue);

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> inputNodeNames = {"data"};
  std::vector<const char *> outputNodeNames = {"features"};

  // Set up the input tensor
  auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value inputTensor =
    Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputNodeDims.data(), 4);
  assert(inputTensor.IsTensor());

  // Feed-forward the model
  auto outputTensors =
    session->Run(Ort::RunOptions{nullptr}, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), 1);
  assert(outputTensors.size() == 1 && outputTensors.front().IsTensor());

  // Get the results back
  float *tensorData = outputTensors.front().GetTensorMutableData<float>();
  cv::Mat currentFeatures(cv::Size(model->outputSize, 1), CV_32F, tensorData);

  // Perform l2 normalizaton
  cv::Mat featuresSquare = currentFeatures.mul(currentFeatures);
  float normalizationValue = sqrt(cv::sum(featuresSquare)[0]);
  currentFeatures = currentFeatures / normalizationValue;
  memcpy(features->data, currentFeatures.data, sizeof(float) * model->outputSize);
}

void buildDatabaseFaceRecognition(const char *databaseFolder, const char *outputPath, FaceRecognitionModelT *model) {
  using namespace boost::filesystem;

  std::vector<std::string> personNames;
  std::vector<int> databaseIds;
  cv::Mat database(cv::Size(model->outputSize, 0), CV_32F);

  path rootPath(databaseFolder);
  if (!exists(rootPath)) {
    std::cerr << "Database path does not exist." << std::endl;
    return;
  }

  int currentId = 0;
  for (auto personPath = directory_iterator(rootPath); personPath != directory_iterator(); personPath++) {
    // For each person in the database
    if (is_directory(personPath->path())) {
      path currentPersonPath(personPath->path());
      personNames.push_back(personPath->path().filename().string());

      for (auto currentImagePath = directory_iterator(currentPersonPath); currentImagePath != directory_iterator();
           currentImagePath++) {
        OpendrImageT image;
        loadImage(currentImagePath->path().string().c_str(), &image);

        cv::Mat features(cv::Size(model->outputSize, 1), CV_32F);
        ffFaceRecognition(model, &image, &features);

        freeImage(&image);
        database.push_back(features.clone());
        databaseIds.push_back(currentId);
      }
      currentId++;
    } else {
      continue;
    }
  }

  if (currentId == 0) {
    std::cerr << "Cannot open database files." << std::endl;
    return;
  }

  // Make the array continuous
  cv::Mat databaseOutput = database.clone();

  std::ofstream fout(outputPath, std::ios::out | std::ios::binary);
  if (!fout.is_open()) {
    std::cerr << "Cannot open database file for writting." << std::endl;
    return;
  }

  // Write number of persons
  int n = personNames.size();

  fout.write(reinterpret_cast<char *>(&n), sizeof(int));
  for (int i = 0; i < n; i++) {
    // Write the name of the person (along with its size)
    int nameLength = personNames[i].size() + 1;
    fout.write(reinterpret_cast<char *>(&nameLength), sizeof(int));
    fout.write(personNames[i].c_str(), nameLength);
  }

  cv::Size s = databaseOutput.size();

  fout.write(reinterpret_cast<char *>(&s.height), sizeof(int));
  fout.write(reinterpret_cast<char *>(&s.width), sizeof(int));
  fout.write(reinterpret_cast<char *>(databaseOutput.data), sizeof(float) * s.height * s.width);
  fout.write(reinterpret_cast<char *>(&databaseIds[0]), sizeof(int) * s.height);
  fout.flush();
  fout.close();
}

void loadDatabaseFaceRecognition(const char *databasePath, FaceRecognitionModelT *model) {
  model->database = NULL;
  model->databaseIds = NULL;

  std::ifstream fin(databasePath, std::ios::out | std::ios::binary);

  if (!fin.is_open()) {
    std::cerr << "Cannot load database file (check that file exists and you have created the database)." << std::endl;
    return;
  }
  int nPerson;
  fin.read(reinterpret_cast<char *>(&nPerson), sizeof(int));
  char **personNames = new char *[nPerson];

  for (int i = 0; i < nPerson; i++) {
    personNames[i] = new char[512];
    // Read person name
    int nameLength;
    fin.read(reinterpret_cast<char *>(&nameLength), sizeof(int));
    if (nameLength > 512) {
      std::cerr << "Person name exceeds max number of characters (512)" << std::endl;
      return;
    }
    fin.read(personNames[i], nameLength);
  }

  int height, width;
  fin.read(reinterpret_cast<char *>(&height), sizeof(int));
  fin.read(reinterpret_cast<char *>(&width), sizeof(int));

  float *databaseBuff = new float[height * width];
  int *featuresIds = new int[height];
  fin.read(reinterpret_cast<char *>(databaseBuff), sizeof(float) * height * width);
  fin.read(reinterpret_cast<char *>(featuresIds), sizeof(int) * height);

  fin.close();

  cv::Mat *database = new cv::Mat(cv::Size(width, height), CV_32F);
  memcpy(database->data, databaseBuff, sizeof(float) * width * height);
  delete[] databaseBuff;

  model->database = database;
  model->databaseIds = featuresIds;
  model->personNames = personNames;
  model->nPersons = nPerson;
  model->nFeatures = height;
}

OpendrCategoryTargetT inferFaceRecognition(FaceRecognitionModelT *model, OpendrImageT *image) {
  cv::Mat features(cv::Size(model->outputSize, 1), CV_32F);
  OpendrCategoryTargetT target;
  target.data = -1;
  target.confidence = 0;

  // Get the feature vector for the current image
  ffFaceRecognition(model, image, &features);

  if (!model->database) {
    std::cerr << "Database is not loaded!" << std::endl;
    return target;
  }
  cv::Mat *database = static_cast<cv::Mat *>(model->database);
  // Calculate the distance between the extracted feature vector and database features
  cv::Mat featuresRepeated;
  cv::repeat(features, model->nFeatures, 1, featuresRepeated);
  cv::Mat differences = featuresRepeated - *database;
  differences = differences.mul(differences);
  cv::Mat squareRootDistances;
  cv::reduce(differences, squareRootDistances, 1, CV_REDUCE_SUM, CV_32F);
  cv::Mat distances;
  cv::sqrt(squareRootDistances, distances);

  double minDistance, maxDistance;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(distances, &minDistance, &maxDistance, &minLoc, &maxLoc);

  target.data = model->databaseIds[minLoc.y];
  target.confidence = 1 - (minDistance / model->threshold);

  return target;
}

void decodeCategoryFaceRecognition(FaceRecognitionModelT *model, OpendrCategoryTargetT category, char *personName) {
  if (category.data >= model->nPersons)
    return;
  strcpy(personName, model->personNames[category.data]);
}

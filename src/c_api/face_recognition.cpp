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
 * Helper function for preprocessing images before feeding them into the face recognition model.
 * This function follows the OpenDR's face recognition pre-processing pipeline, which includes the following:
 * a) resizing the image into resize_target x resize_target pixels and then taking a center crop of size model_input_size,
 * and b) normalizing the resulting values using mean_value and std_value
 * @param image image to be preprocesses
 * @param data pre-processed data in a flattened vector
 * @param resize_target target size for resizing
 * @param model_input_size size of the center crop (equals the size that the DL model expects)
 * @param mean_value value used for centering the input image
 * @param std_value value used for scaling the input image
 */
void preprocess_face_recognition(cv::Mat *image, std::vector<float> &data, int resize_target = 128, int model_input_size = 112,
                                 float mean_value = 0.5, float std_value = 0.5) {
  // Convert to RGB
  cv::Mat img;
  cv::cvtColor(*image, img, cv::COLOR_BGR2RGB);

  // Resize and then get a center crop
  cv::resize(img, img, cv::Size(resize_target, resize_target));
  int stride = (resize_target - model_input_size) / 2;
  cv::Rect myROI(stride, stride, resize_target - stride, resize_target - stride);
  img = img(myROI);

  // Scale to 0...1
  cv::Mat out_img;
  img.convertTo(out_img, CV_32FC3, 1 / 255.0);
  // Unfold the image into the appropriate format
  // This is certainly not the most efficient way to do this...
  // ... and is probably constantly leading to cache misses
  // ... but it works for now.
  for (unsigned int j = 0; j < model_input_size; ++j) {
    for (unsigned int k = 0; k < model_input_size; ++k) {
      cv::Vec3f cur_pixel = out_img.at<cv::Vec3f>(j, k);
      data[0 * model_input_size * model_input_size + j * model_input_size + k] = (cur_pixel[0] - mean_value) / std_value;
      data[1 * model_input_size * model_input_size + j * model_input_size + k] = (cur_pixel[1] - mean_value) / std_value;
      data[2 * model_input_size * model_input_size + j * model_input_size + k] = (cur_pixel[2] - mean_value) / std_value;
    }
  }
}

/**
 * Very simple helper function to parse OpenDR model files for face recognition
 * In the future this can be done at library level using a JSON-parser
 */
std::string json_get_key_string(std::string json, const std::string &key) {
  std::size_t start_idx = json.find(key);
  std::string value = json.substr(start_idx);
  value = value.substr(value.find(":") + 1);
  value.resize(value.find(","));
  value = value.substr(value.find("\"") + 1);
  value.resize(value.find("\""));
  return value;
}

void load_face_recognition_model(const char *model_path, face_recognition_model_t *model) {
  // Initialize model
  model->onnx_session = model->env = model->session_options = NULL;
  model->database = model->database_ids = NULL;
  model->person_names = NULL;
  model->threshold = 1;

  // Parse the model JSON file
  std::string model_json_path(model_path);
  std::size_t split_pos = model_json_path.find_last_of("/");
  split_pos = split_pos > 0 ? split_pos + 1 : 0;
  model_json_path = model_json_path + "/" + model_json_path.substr(split_pos) + ".json";

  std::ifstream in_stream(model_json_path);
  if (!in_stream.is_open()) {
    std::cerr << "Cannot open JSON model file" << std::endl;
    return;
  }

  std::string str;
  in_stream.seekg(0, std::ios::end);
  str.reserve(in_stream.tellg());
  in_stream.seekg(0, std::ios::beg);
  str.assign((std::istreambuf_iterator<char>(in_stream)), std::istreambuf_iterator<char>());

  std::string basepath = model_json_path.substr(0, split_pos);
  split_pos = basepath.find_last_of("/");
  split_pos = split_pos > 0 ? split_pos + 1 : 0;
  basepath.resize(split_pos);

  // Parse JSON
  std::string onnx_model_path = basepath + json_get_key_string(str, "model_paths");
  std::string model_format = json_get_key_string(str, "format");

  // Parse inference params
  std::string threshold = json_get_key_string(str, "threshold");
  ;
  if (!threshold.empty()) {
    model->threshold = std::stof(threshold);
  }

  // Proceed only if the model is in onnx format
  if (model_format != "onnx") {
    std::cerr << "Model not in ONNX format." << std::endl;
    return;
  }

  Ort::Env *env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "opendr_env");

  Ort::SessionOptions *session_options = new Ort::SessionOptions;
  session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  Ort::Session *session = new Ort::Session(*env, onnx_model_path.c_str(), *session_options);

  model->env = env;
  model->onnx_session = session;
  model->session_options = session_options;

  // Should we pass these parameters through the model json file?
  model->model_size = 112;
  model->resize_size = 128;
  model->mean_value = 0.5;
  model->std_value = 0.5;
  model->output_size = 128;
}

void free_face_recognition_model(face_recognition_model_t *model) {
  if (model->onnx_session) {
    Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);
    delete session;
  }

  if (model->session_options) {
    Ort::SessionOptions *session_options = static_cast<Ort::SessionOptions *>(model->session_options);
    delete session_options;
  }

  if (model->env) {
    Ort::Env *env = static_cast<Ort::Env *>(model->env);
    delete env;
  }

  if (model->database_ids) {
    delete[] model->database_ids;
  }

  if (model->database) {
    cv::Mat *database = static_cast<cv::Mat *>(model->database);
    delete database;
  }

  if (model->person_names) {
    for (int i = 0; i < model->n_persons; i++)
      delete[] model->person_names[i];
    delete[] model->person_names;
  }
}

void ff_face_recognition(face_recognition_model_t *model, opendr_image_t *image, cv::Mat *features) {
  Ort::Session *session = static_cast<Ort::Session *>(model->onnx_session);
  if (!session) {
    std::cerr << "ONNX session not initialized." << std::endl;
    return;
  }

  // Prepare the input dimensions
  std::vector<int64_t> input_node_dims = {1, 3, model->model_size, model->model_size};
  size_t input_tensor_size = model->model_size * model->model_size * 3;

  // Get the input image and pre-process it
  std::vector<float> input_tensor_values(input_tensor_size);
  cv::Mat *opencv_image = static_cast<cv::Mat *>(image->data);
  if (!opencv_image) {
    std::cerr << "Cannot load image for inference." << std::endl;
    return;
  }

  preprocess_face_recognition(opencv_image, input_tensor_values, model->resize_size, model->model_size, model->mean_value,
                              model->std_value);

  // Setup input/output names
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<const char *> input_node_names = {"data"};
  std::vector<const char *> output_node_names = {"features"};

  // Setup the input tensor
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor =
    Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // Feed-forward the model
  auto output_tensors =
    session->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get the results back
  float *floatarr = output_tensors.front().GetTensorMutableData<float>();
  cv::Mat cur_features(cv::Size(model->output_size, 1), CV_32F, floatarr);

  // Perform l2 normalizaton
  cv::Mat features_square = cur_features.mul(cur_features);
  float norm = sqrt(cv::sum(features_square)[0]);
  cur_features = cur_features / norm;
  memcpy(features->data, cur_features.data, sizeof(float) * model->output_size);
}

void build_database_face_recognition(const char *database_folder, const char *output_path, face_recognition_model_t *model) {
  using namespace boost::filesystem;

  std::vector<std::string> person_names;
  std::vector<int> database_ids;
  cv::Mat database(cv::Size(model->output_size, 0), CV_32F);

  path root_path(database_folder);
  if (!exists(root_path)) {
    std::cerr << "Database path does not exist." << std::endl;
    return;
  }

  int current_id = 0;
  for (auto person_path = directory_iterator(root_path); person_path != directory_iterator(); person_path++) {
    // For each person in the database
    if (is_directory(person_path->path())) {
      path cur_person_path(person_path->path());
      person_names.push_back(person_path->path().filename().string());

      for (auto cur_img_path = directory_iterator(cur_person_path); cur_img_path != directory_iterator(); cur_img_path++) {
        opendr_image_t image;
        load_image(cur_img_path->path().string().c_str(), &image);

        cv::Mat features(cv::Size(model->output_size, 1), CV_32F);
        ff_face_recognition(model, &image, &features);

        free_image(&image);
        database.push_back(features.clone());
        database_ids.push_back(current_id);
      }
      current_id++;
    } else {
      continue;
    }
  }

  if (current_id == 0) {
    std::cerr << "Cannot open database files." << std::endl;
    return;
  }

  // Make the array continuous
  cv::Mat database_out = database.clone();

  std::ofstream fout(output_path, std::ios::out | std::ios::binary);
  if (!fout.is_open()) {
    std::cerr << "Cannot open database file for writting." << std::endl;
    return;
  }

  // Write number of persons
  int n = person_names.size();

  fout.write(reinterpret_cast<char *>(&n), sizeof(int));
  for (int i = 0; i < n; i++) {
    // Write the name of the person (along with its size)
    int name_length = person_names[i].size() + 1;
    fout.write(reinterpret_cast<char *>(&name_length), sizeof(int));
    fout.write(person_names[i].c_str(), name_length);
  }

  cv::Size s = database_out.size();

  fout.write(reinterpret_cast<char *>(&s.height), sizeof(int));
  fout.write(reinterpret_cast<char *>(&s.width), sizeof(int));
  fout.write(reinterpret_cast<char *>(database_out.data), sizeof(float) * s.height * s.width);
  fout.write(reinterpret_cast<char *>(&database_ids[0]), sizeof(int) * s.height);
  fout.flush();
  fout.close();
}

void load_database_face_recognition(const char *database_path, face_recognition_model_t *model) {
  model->database = NULL;
  model->database_ids = NULL;

  std::ifstream fin(database_path, std::ios::out | std::ios::binary);

  if (!fin.is_open()) {
    std::cerr << "Cannot load database file (check that file exists and you have created the database)." << std::endl;
    return;
  }
  int n;
  fin.read(reinterpret_cast<char *>(&n), sizeof(int));
  char **person_names = new char *[n];

  for (int i = 0; i < n; i++) {
    person_names[i] = new char[512];
    // Read person name
    int name_length;
    fin.read(reinterpret_cast<char *>(&name_length), sizeof(int));
    if (name_length > 512) {
      std::cerr << "Person name exceeds max number of characters (512)" << std::endl;
      return;
    }
    fin.read(person_names[i], name_length);
  }

  int height, width;
  fin.read(reinterpret_cast<char *>(&height), sizeof(int));
  fin.read(reinterpret_cast<char *>(&width), sizeof(int));

  float *database_buff = new float[height * width];
  int *features_ids = new int[height];
  fin.read(reinterpret_cast<char *>(database_buff), sizeof(float) * height * width);
  fin.read(reinterpret_cast<char *>(features_ids), sizeof(int) * height);

  fin.close();

  cv::Mat *database = new cv::Mat(cv::Size(width, height), CV_32F);
  memcpy(database->data, database_buff, sizeof(float) * width * height);
  delete[] database_buff;

  model->database = database;
  model->database_ids = features_ids;
  model->person_names = person_names;
  model->n_persons = n;
  model->n_features = height;
}

opendr_category_target_t infer_face_recognition(face_recognition_model_t *model, opendr_image_t *image) {
  cv::Mat features(cv::Size(model->output_size, 1), CV_32F);
  opendr_category_target_t target;
  target.data = -1;
  target.confidence = 0;

  // Get the feature vector for the current image
  ff_face_recognition(model, image, &features);

  if (!model->database) {
    std::cerr << "Database is not loaded!" << std::endl;
    return target;
  }
  cv::Mat *database = static_cast<cv::Mat *>(model->database);
  // Calculate the distance between the extracted feature vector and database features
  cv::Mat features_repeated;
  cv::repeat(features, model->n_features, 1, features_repeated);
  cv::Mat diff = features_repeated - *database;
  diff = diff.mul(diff);
  cv::Mat sq_dists;
  cv::reduce(diff, sq_dists, 1, CV_REDUCE_SUM, CV_32F);
  cv::Mat dists;
  cv::sqrt(sq_dists, dists);

  double min_dist, max_dist;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc(dists, &min_dist, &max_dist, &min_loc, &max_loc);

  target.data = model->database_ids[min_loc.y];
  target.confidence = 1 - (min_dist / model->threshold);

  return target;
}

void decode_category_face_recognition(face_recognition_model_t *model, opendr_category_target_t category, char *person_name) {
  if (category.data >= model->n_persons)
    return;
  strcpy(person_name, model->person_names[category.data]);
}

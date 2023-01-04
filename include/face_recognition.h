/*
 * Copyright 2020-2023 OpenDR European Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef C_API_FACE_RECOGNITION_H
#define C_API_FACE_RECOGNITION_H

#include "opendr_utils.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

struct face_recognition_model {
  // ONNX session objects
  void *onnx_session;
  void *env;
  void *session_options;

  // Sizes for resizing and cropping an input image
  int model_size;
  int resize_size;

  // Statistics for normalization
  float mean_value;
  float std_value;

  // Recognition threshold
  float threshold;

  // Feature dimension
  int output_size;

  // Database data
  void *database;
  int *database_ids;
  char **person_names;

  // Number of persons in the database
  int n_persons;
  // Number of features vectors in the database
  int n_features;
};
typedef struct face_recognition_model face_recognition_model_t;

/**
 * Loads a face recognition model saved in OpenDR format
 * @param model_path path to the OpenDR face recongition model (as exported using OpenDR library)
 * @param model the loaded model
 */
void load_face_recognition_model(const char *model_path, face_recognition_model_t *model);

/**
 * This function perform inference using a face recognition model and an input image
 * @param model face recognition model to be used for inference
 * @param image OpenDR image
 * @return OpenDR classification target containing the id of the recognized person
 */
opendr_category_target_t infer_face_recognition(face_recognition_model_t *model, opendr_image_t *image);

/**
 * Builds a face recognition database (containing images for persons to be recognized). This function expects the
 * database_folder to have the same format as the main Python toolkit.
 * @param database_folder folder containing the database
 * @param output_path output path to store the binary database. This file should be loaded along with the face
 * recognition model before performing inference.
 * @param model the face recognition model to be used for extracting the database features
 */
void build_database_face_recognition(const char *database_folder, const char *output_path, face_recognition_model_t *model);

/**
 * Loads an already built database into the face recognition model. After this step, the model can be used for
 * performing inference.
 * @param database_path path to the database file
 * @param model the face recognition model to be used for inference
 */
void load_database_face_recognition(const char *database_path, face_recognition_model_t *model);

/**
 * Returns the name of a recognition person by decoding the category id into a string
 * @param model the face recognition model to be used for inference
 * @param category the predicted category
 * @param person_name buffer to store the person name
 */
void decode_category_face_recognition(face_recognition_model_t *model, opendr_category_target_t category, char *person_name);

/**
 * Releases the memory allocated for a face recognition model
 * @param model model to be de-allocated
 */
void free_face_recognition_model(face_recognition_model_t *model);

#ifdef __cplusplus
}
#endif

#endif  // C_API_FACE_RECOGNITION_H

/*
 * Copyright 2020-2022 OpenDR European Project
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

#ifndef C_API_OPENDR_UTILS_H
#define C_API_OPENDR_UTILS_H

#include "data.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Json parser for OpenDR model files.
 * @param json a string of json file.
 * @param key the value to extract from json file.
 * @param index the index to choose the value if it is an array.
 * @return string with the value of key
 */
const char *json_get_key_string(const char *json, const char *key, const int index);

/**
 * Json parser for OpenDR model files.
 * @param json a string of json file.
 * @param key the value to extract from json file.
 * @param index the index to choose the value if it is an array.
 * @return float with the value of key
 */
float json_get_key_float(const char *json, const char *key, const int index);

/**
 * Json parser for OpenDR model files from inference_params key.
 * @param json a string of json file.
 * @param key the value to extract from inference_params.
 * @param index the index to choose the value if it is an array.
 * @return float with the value of key
 */
float json_get_key_from_inference_params(const char *json, const char *key, const int index);

/**
 * Reads an image from path and saves it into OpenDR an image structure
 * @param path path from which the image will be read
 * @param image OpenDR image data structure to store the image
 */
void load_image(const char *path, opendr_image_t *image);

/**
 * Releases the memory allocated for an OpenDR image structure
 * @param image OpenDR image structure to release
 */
void free_image(opendr_image_t *image);

/**
 * Initialize an empty Opendr detection vector target to be used in C API
 * @param detection_vector OpenDR detection_target_list structure to be initialized
 */
void init_detections_vector(opendr_detection_vector_target_t *detection_vector);

/**
 * Loads an OpenDR detection target list to be used in C API
 * @param detection_vector OpenDR detection_target_list structure to be loaded
 * @param detection the pointer of the first OpenDR detection target in a vector
 * @param vector_size the size of the vector
 */
void load_detections_vector(opendr_detection_vector_target_t *detection_vector, opendr_detection_target_t *detection,
                            int vector_size);

/**
 * Releases the memory allocated for a detection list structure
 * @param detection_vector OpenDR detection vector target structure to release
 */
void free_detections_vector(opendr_detection_vector_target_t *detection_vector);

/**
 * Initialize an empty OpenDR tensor to be used in C API
 * @param tensor OpenDR tensor to initialize
 */
void init_tensor(opendr_tensor_t *opendr_tensor);

/**
 * Load a tensor values to be used in C.
 * @param tensor Opendr tensor structure to be loaded
 * @param tensor_data vector of cv Mat that holds tensors data to be used
 * @param batch_size batch size for each OpenDR mat in an array of integers
 * @param frames frames size for each OpenDR mat in an array of integers
 * @param channels channels size for each OpenDR mat in an array of integers
 * @param width width size for each OpenDR mat in an array of integers
 * @param height height size for each OpenDR mat in an array of integers
 */
void load_tensor(opendr_tensor_t *opendr_tensor, void *tensor_data, int batch_size, int frames, int channels, int width,
                 int height);

/**
 * Releases the memory allocated for an OpenDR tensor structure
 * @param opendr_tensor OpenDR tensor structure to release
 */
void free_tensor(opendr_tensor_t *opendr_tensor);

/**
 * Initialize an empty OpenDR tensor vector to be used in C API
 * @param tensor_vector
 */
void init_tensor_vector(opendr_tensor_vector_t *tensor_vector);

/**
 * Load a vector of tensors values to be used in C.
 * @param tensor_vector OpenDR vector of tensors structure to be loaded
 * @param tensor data in vector of OpenDR tensors structure
 * @param number_of_tensors the number of tensors that we want to load into the structure
 */
void load_tensor_vector(opendr_tensor_vector_t *tensor_vector, opendr_tensor_t *tensor, int number_of_tensors);

/**
 * Releases the memory allocated for an OpenDR tensor vector structure
 * @param tensor_vector OpenDR tensor vector structure to release
 */
void free_tensor_vector(opendr_tensor_vector_t *tensor_vector);

/**
 * Helper function to store a tensor from OpenDR tensor vector structure into an OpenCV Mat.
 * @param tensor OpenDR tensor to store the tensor
 * @param tensor_vector OpenDR tensor vector from which tensor is wanted
 * @param index the tensor is wanted from Opendr tensor vector
 */
void iter_tensor_vector(opendr_tensor_t *output, opendr_tensor_vector_t *tensor_vector, int index);

#ifdef __cplusplus
}
#endif

#endif  // C_API_OPENDR_UTILS_H

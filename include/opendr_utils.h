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
 */
const char *json_get_key_string(const char *json, const char *key);

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
 * Initialize an empty detection list to be used in C API
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

#ifdef __cplusplus
}
#endif

#endif  // C_API_OPENDR_UTILS_H

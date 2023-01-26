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

#ifndef C_API_OPENDR_UTILS_H
#define C_API_OPENDR_UTILS_H

#include "data.h"
#include "target.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * JSON parser to extract strings from OpenDR model files.
 * @param json a string of json file
 * @param key the key of value to extract from json file
 * @param index the index to choose the value if it is an array, otherwise it is not be used
 * @return string with the value of the key
 */
const char *jsonGetStringFromKey(const char *json, const char *key, const int index);

/**
 * JSON parser to extract floats from OpenDR model files.
 * @param json a string of json file
 * @param key the key of value to extract from json file
 * @param index the index to choose the value if it is an array, otherwise it is not be used
 * @return float with the value of the key
 */
float jsonGetFloatFromKey(const char *json, const char *key, const int index);

/**
 * JSON parser to extract strings from OpenDR model files inference_params key.
 * @param json a string of json file
 * @param key the key of value to extract from inference_params
 * @param index the index to choose the value if it is an array, otherwise it is not be used
 * @return string with the value of the key
 */
const char *jsonGetStringFromKeyInInferenceParams(const char *json, const char *key, const int index);

/**
 * JSON parser to extract floats from OpenDR model files inference_params key.
 * @param json a string of json file
 * @param key the value to extract from inference_params
 * @param index the index to choose the value if it is an array, otherwise it is not be used
 * @return float with the value of the key
 */
float jsonGetFloatFromKeyInInferenceParams(const char *json, const char *key, const int index);

/**
 * Reads an image from path and saves it into OpenDR image structure.
 * @param path path from which the image will be read
 * @param image OpenDR image data structure to store the image
 */
void loadImage(const char *path, OpendrImageT *image);

/**
 * Releases the memory allocated for an OpenDR image structure
 * @param image OpenDR image structure to release
 */
void freeImage(OpendrImageT *image);

/**
 * Initialize an empty detection list.
 * @param vector OpenDR OpendrDetectionVectorTargetT structure to be initialized
 */
void initDetectionsVector(OpendrDetectionVectorTargetT *vector);

/**
 * Loads an OpenDR vector of detections.
 * @param vector OpenDR OpendrDetectionVectorTargetT structure to be loaded
 * @param detectionPtr the pointer of the first OpenDR detection target in a vector
 * @param vectorSize the size of the vector
 */
void loadDetectionsVector(OpendrDetectionVectorTargetT *vector, OpendrDetectionTargetT *detectionPtr, int vectorSize);

/**
 * Releases the memory allocated for a vector of detections
 * @param vector OpenDR vector of detections structure to release
 */
void freeDetectionsVector(OpendrDetectionVectorTargetT *vector);

/**
 * Initialize an empty OpenDR tensor
 * @param tensor OpenDR tensor to initialize
 */
void initTensor(OpendrTensorT *tensor);

/**
 * Load a OpenDR tensor
 * @param tensor OpenDR tensor structure to be loaded
 * @param tensorData data pointer of a vector that holds tensors data to be loaded
 * @param batchSize batch size of tensor
 * @param frames frames size of tensor
 * @param channels channels size of tensor
 * @param width width size of tensor
 * @param height height size of tensor
 */
void loadTensor(OpendrTensorT *tensor, void *tensorData, int batchSize, int frames, int channels, int width, int height);

/**
 * Releases the memory allocated for an OpenDR tensor structure
 * @param tensor OpenDR tensor structure to release
 */
void freeTensor(OpendrTensorT *tensor);

/**
 * Initialize an empty OpenDR vector of tensors
 * @param vector an OpenDR vector of tensors to initialize
 */
void initTensorVector(OpendrTensorVectorT *vector);

/**
 * Load an OpenDR vector of tensors
 * @param vector OpenDR vector of tensors structure to be loaded
 * @param tensorPtr pointer of a vector of OpenDR tensors structure
 * @param nTensors the number of tensors that we want to load into the structure
 */
void loadTensorVector(OpendrTensorVectorT *vector, OpendrTensorT *tensorPtr, int nTensors);

/**
 * Releases the memory allocated for an OpenDR vector of tensors
 * @param vector OpenDR vector of tensors structure to release
 */
void freeTensorVector(OpendrTensorVectorT *vector);

/**
 * Helper function to store an OpenDR tensor from a specific tensor of OpenDR vector of tensors
 * @param tensor OpenDR tensor to store the wanted data
 * @param vector the OpenDR vector of tensors source of wanted tensor
 * @param index the index of tensor that is wanted from OpenDR vector
 */
void iterTensorVector(OpendrTensorT *tensor, OpendrTensorVectorT *vector, int index);

#ifdef __cplusplus
}
#endif

#endif  // C_API_OPENDR_UTILS_H

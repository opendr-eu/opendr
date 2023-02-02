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

#include <stdio.h>
#include <stdlib.h>
#include "opendr_utils.h"
#include "skeleton_based_action_recognition_pst.h"

int main(int argc, char **argv) {
  PstModelT model;

  printf("start init model\n");
  loadPstModel("data/skeleton_based_action_recognition/progressive_spatiotemporal_gcn/optimized_model/onnx_model.onnx", &model);
  printf("success\n");

  // Initialize OpenDR tensor for input
  OpenDRTensorT input_tensor;
  initTensor(&input_tensor);

  initRandomOpenDRTensorPst(&input_tensor, &model);

  // Initialize OpenDR tensor vector for output
  OpenDRTensorVectorT output_tensor_vector;
  initTensorVector(&output_tensor_vector);

  forwardPst(&model, &input_tensor, &output_tensor_vector);

  // Free the memory
  freeTensor(&input_tensor);
  freeTensorVector(&output_tensor_vector);
  freePstModel(&model);

  return 0;
}

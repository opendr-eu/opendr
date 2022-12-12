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

#include <stdio.h>
#include <stdlib.h>
#include "object_tracking_2d_deep_sort.h"
#include "opendr_utils.h"

int main(int argc, char **argv) {
  deep_sort_model_t model;

  printf("start init model\n");
  load_deep_sort_model("data/object_tracking_2d/deep_sort/optimized_model/onnx_model.onnx", &model);
  printf("success\n");

  // Initialize opendr tensor for input
  opendr_tensor_t input_tensor;
  init_random_opendr_tensor_ds(&input_tensor, &model);

  // Initialize opendr tensor vector for output
  opendr_tensor_vector_t output_tensor_vector;
  forward_deep_sort(&model, &input_tensor, &output_tensor_vector);

  // Free the memory
  free_tensor(&input_tensor);
  free_tensor_vector(&output_tensor_vector);
  free_deep_sort_model(&model);

  return 0;
}

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
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "opendr_utils.h"
#include "skeleton_based_action_recognition_pst.h"

START_TEST(model_creation_test) {
  // Create a skeleton based action recognition pst model
  pst_model_t model;

  // Load a pretrained model
  load_pst_model("data/skeleton_based_action_recognition/progressive_spatiotemporal_gcn/optimized_model/onnx_model.onnx",
                 &model);

  ck_assert(model.onnx_session);
  ck_assert(model.env);
  ck_assert(model.session_options);

  free_pst_model(&model);
}
END_TEST

START_TEST(forward_pass_creation_test) {
  // Create a skeleton based action recognition pst model
  pst_model_t model;
  // Load a pretrained model (see instructions for downloading the data)
  load_pst_model("data/skeleton_based_action_recognition/progressive_spatiotemporal_gcn/optimized_model/onnx_model.onnx",
                 &model);
  // Load a random tensor and perform forward pass
  opendr_tensor_t input_tensor;
  init_tensor(&input_tensor);

  init_random_opendr_tensor_pst(&input_tensor, &model);

  // Initialize opendr tensor vector for output
  opendr_tensor_vector_t output_tensor_vector;
  init_tensor_vector(&output_tensor_vector);

  forward_pst(&model, &input_tensor, &output_tensor_vector);

  // Load another tensor
  init_random_opendr_tensor_pst(&input_tensor, &model);
  forward_pst(&model, &input_tensor, &output_tensor_vector);

  ck_assert(output_tensor_vector.n_tensors == 1);

  free_pst_model(&model);
  free_tensor(&input_tensor);
  free_tensor_vector(&output_tensor_vector);
}
END_TEST

Suite *pst_gcn_suite(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("Pst Gcn");
  tc_core = tcase_create("Core");

  tcase_set_timeout(tc_core, 60.0);
  tcase_add_test(tc_core, model_creation_test);
  tcase_add_test(tc_core, forward_pass_creation_test);
  suite_add_tcase(s, tc_core);

  return s;
}

int main() {
  int no_failed = 0;
  Suite *s;
  SRunner *runner;

  s = pst_gcn_suite();
  runner = srunner_create(s);

  srunner_run_all(runner, CK_NORMAL);
  no_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return (no_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

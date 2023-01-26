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
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "lightweight_open_pose.h"
#include "opendr_utils.h"

START_TEST(model_creation_test) {
  // Create a face recognition model
  OpenPoseModelT model;

  // Load a pretrained model
  loadOpenPoseModel("data/lightweight_open_pose/optimized_model/onnx_model.onnx", &model);

  ck_assert(model.onnxSession);
  ck_assert(model.env);
  ck_assert(model.sessionOptions);

  freeOpenPoseModel(&model);
}
END_TEST

START_TEST(forward_pass_creation_test) {
  // Create a x3d model
  OpenPoseModelT model;
  // Load a pretrained model (see instructions for downloading the data)
  loadOpenPoseModel("data/lightweight_open_pose/optimized_model/onnx_model.onnx", &model);

  // Load a random tensor and perform forward pass
  OpendrTensorT input_tensor;
  initTensor(&input_tensor);

  initRandomOpendrTensorOp(&input_tensor, &model);

  // Initialize opendr tensor vector for output
  OpendrTensorVectorT output_tensor_vector;
  initTensorVector(&output_tensor_vector);

  forwardOpenPose(&model, &input_tensor, &output_tensor_vector);

  // Load another tensor
  initRandomOpendrTensorOp(&input_tensor, &model);
  forwardOpenPose(&model, &input_tensor, &output_tensor_vector);

  ck_assert(output_tensor_vector.nTensors == model.outputSize);

  // Free the model resources
  freeOpenPoseModel(&model);
  freeTensor(&input_tensor);
  freeTensorVector(&output_tensor_vector);
}
END_TEST

Suite *open_pose_suite(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("Open Pose");
  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, model_creation_test);
  tcase_add_test(tc_core, forward_pass_creation_test);
  suite_add_tcase(s, tc_core);

  return s;
}

int main() {
  int no_failed = 0;
  Suite *s;
  SRunner *runner;

  s = open_pose_suite();
  runner = srunner_create(s);

  srunner_run_all(runner, CK_NORMAL);
  no_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return (no_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

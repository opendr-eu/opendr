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
#include "face_recognition.h"
#include "opendr_utils.h"

START_TEST(model_creation_test) {
  // Create a face recognition model
  FaceRecognitionModelT model;

  // Load a pretrained model
  loadFaceRecognitionModel("data/optimized_model", &model);

  ck_assert(model.onnxSession);
  ck_assert(model.env);
  ck_assert(model.sessionOptions);

  // Release the resources
  freeFaceRecognitionModel(&model);

  // Load a model that does not exist
  loadFaceRecognitionModel("data/optimized_model_not_existant", &model);
  ck_assert(!model.onnxSession);
  ck_assert(!model.env);
  ck_assert(!model.sessionOptions);

  // Release the resources
  freeFaceRecognitionModel(&model);
}
END_TEST

START_TEST(database_creation_test) {
  FaceRecognitionModelT model;
  loadFaceRecognitionModel("data/optimized_model", &model);

  // Check that we can create and load a database that exists
  buildDatabaseFaceRecognition("data/database", "data/database.dat", &model);
  loadDatabaseFaceRecognition("data/database.dat", &model);
  ck_assert(model.database);
  ck_assert(model.databaseIds);
  ck_assert(model.databaseIds);

  // Check that we can handle errors in the process
  buildDatabaseFaceRecognition("data/database_not_existant", "data/database.dat", &model);
  loadDatabaseFaceRecognition("data/database_not_existant.dat", &model);
  ck_assert(!model.database);
  ck_assert(!model.databaseIds);

  // Release the resources
  freeFaceRecognitionModel(&model);
}
END_TEST

START_TEST(inference_creation_test) {
  // Create a face recognition model
  FaceRecognitionModelT model;
  // Load a pretrained model (see instructions for downloading the data)
  loadFaceRecognitionModel("data/optimized_model", &model);

  // Build and load the database
  buildDatabaseFaceRecognition("data/database", "data/database.dat", &model);
  loadDatabaseFaceRecognition("data/database.dat", &model);

  // Load an image and performance inference
  OpendrImageT image;
  loadImage("data/database/1/1.jpg", &image);
  OpendrCategoryTargetT res = inferFaceRecognition(&model, &image);
  freeImage(&image);
  char buff[512];
  decodeCategoryFaceRecognition(&model, res, buff);
  ck_assert(!strcmp(buff, "1"));

  // Load another image
  loadImage("data/database/5/1.jpg", &image);
  res = inferFaceRecognition(&model, &image);
  freeImage(&image);
  decodeCategoryFaceRecognition(&model, res, buff);
  ck_assert(!strcmp(buff, "5"));

  // Free the model resources
  freeFaceRecognitionModel(&model);
}
END_TEST

Suite *face_recognition_suite(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("Face Recognition");
  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, model_creation_test);
  tcase_add_test(tc_core, database_creation_test);
  tcase_add_test(tc_core, inference_creation_test);
  suite_add_tcase(s, tc_core);

  return s;
}

int main() {
  int no_failed = 0;
  Suite *s;
  SRunner *runner;

  s = face_recognition_suite();
  runner = srunner_create(s);

  srunner_run_all(runner, CK_NORMAL);
  no_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return (no_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

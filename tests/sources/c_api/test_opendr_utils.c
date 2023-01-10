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
#include <assert.h>
#include <check.h>
#include <stdio.h>
#include <stdlib.h>
#include "opendr_utils.h"

START_TEST(image_load_test) {
  // Load an image and performance inference
  opendr_image_t image;
  // An example of an image that exist
  load_image("data/database/1/1.jpg", &image);
  ck_assert(image.data);
  // An example of an image that does not exist
  load_image("images/not_existant/1.jpg", &image);
  ck_assert(image.data == 0);

  // Free the resources
  free_image(&image);
}
END_TEST

Suite *opendr_utilities_suite(void) {
  Suite *s;
  TCase *tc_core;

  s = suite_create("OpenDR Utilities");
  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, image_load_test);
  suite_add_tcase(s, tc_core);

  return s;
}

int main() {
  int no_failed = 0;
  Suite *s;
  SRunner *runner;

  s = opendr_utilities_suite();
  runner = srunner_create(s);

  srunner_run_all(runner, CK_NORMAL);
  no_failed = srunner_ntests_failed(runner);
  srunner_free(runner);
  return (no_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
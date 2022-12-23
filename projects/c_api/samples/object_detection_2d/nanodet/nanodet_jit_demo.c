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
#include "object_detection_2d_nanodet_jit.h"
#include "opendr_utils.h"

int main(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr,
            "usage: %s [model_path] [device] [images_path] [input_sizes].\n"
            "model_path = path/to/your/libtorch/model.pth \ndevice = cuda or cpu \n"
            "images_path = \"xxx/xxx/*.jpg\" \ninput_size = width height.\n",
            argv[0]);
    return -1;
  }

  nanodet_model_t model;

  int height = atoi(argv[4]);
  int width = atoi(argv[5]);
  printf("start init model\n");
  load_nanodet_model(argv[1], argv[2], height, width, 0.35, &model);
  printf("success\n");

  opendr_image_t image;

  load_image(argv[3], &image);
  if (!image.data) {
    printf("Image not found!");
    return 1;
  }

  // Initialize opendr detection target list;
  opendr_detection_vector_target_t results;
  init_detections_vector(&results);

  results = infer_nanodet(&model, &image);

  draw_bboxes(&image, &model, &results);

  // Free the memory
  free_detections_vector(&results);
  free_image(&image);
  free_nanodet_model(&model);

  return 0;
}

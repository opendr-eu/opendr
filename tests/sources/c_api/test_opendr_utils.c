/*
 * Copyright 2020-2021 OpenDR project
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
#include <stdio.h>
#include <stdlib.h>
#include "opendr_utils.h"

void image_load_test() {
    // Load an image and performance inference
    opendr_image_t image;
    // An example of an image that exist
    load_image("data/database/1/1.jpg", &image);
    assert(image.data);
    // An example of an image that does not exist
    load_image("images/not_existant/1.jpg", &image);
    assert(image.data == 0);

    // Free the resources
    free_image(&image);
}

int main(int argc, char *argv[]) {
    image_load_test();
    return 0;
}

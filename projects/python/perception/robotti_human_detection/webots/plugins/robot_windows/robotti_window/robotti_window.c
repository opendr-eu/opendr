/*
 * Copyright 2020-2024 OpenDR European Project
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

#include <webots/camera.h>
#include <webots/plugins/robot_window/robot_wwi.h>
#include <webots/robot.h>
#include <webots/supervisor.h>

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double FIELD_SIZE[2] = {40, 14};
static double FIELD_TRANSLATION[2] = {-20, 0};
static WbFieldRef root_children_field = NULL;
static WbDeviceTag display = 0;
static WbDeviceTag camera = 0;
static int i = 1;

// Window initialization: get some robot devices.
void wb_robot_window_init() {
  WbNodeRef root_node = wb_supervisor_node_get_root();
  root_children_field = wb_supervisor_node_get_field(root_node, "children");
  display = wb_robot_get_device("recognition display");
  camera = wb_robot_get_device("front_bottom_camera");
  srand((unsigned int)time(NULL));
}

// A simulation step occurred.
void wb_robot_window_step(int time_step) {
  // Window initialization: get some robot devices.
  const char *message;
  while ((message = wb_robot_wwi_receive_text())) {
    printf("%s\n", message);
    if (strcmp(message, "shot") == 0) {
      char file_name[128];
      sprintf(file_name, "image_%d.jpeg", i);
      printf("%s\n", file_name);
      wb_camera_save_image(camera, file_name, 100);
      i++;

    } else if (strncmp(message, "add obstacle", strlen("add obstacle")) == 0) {
      // Add new obstacle
      char class[64], def_name[64];
      float x, y;
      const int found = sscanf(message, "add obstacle %s %s %f %f", class, def_name, &x, &y);
      if (found != 2 && found != 4) {
        fprintf(stderr, "Invalid instructions to add obstacle.\n");
        continue;
      }
      if (found == 2) {
        // random position
        const double rx = (double)rand() / (double)(RAND_MAX);
        const double ry = (double)rand() / (double)(RAND_MAX);
        x = (rx - 0.5) * FIELD_SIZE[0] + FIELD_TRANSLATION[0];
        y = (ry - 0.5) * FIELD_SIZE[1] + FIELD_TRANSLATION[1];
      }
      // random orientation
      const double angle = (double)rand() / (double)(RAND_MAX)*2 * M_PI;
      char new_node_string[200];
      char optional_parameters[32] = "";
      for (int i = 0; i < strlen(class); i++)
        class[i] = tolower(class[i]);
      if (strcmp(class, "horse") == 0)
        strcpy(optional_parameters, "colorBody 0.388 0.271 0.173 ");
      class[0] = toupper(class[0]);
      sprintf(new_node_string, "DEF %s %s { translation %f %f 0.05 rotation 0 0 1 %f %s}", def_name, class, x, y, angle,
              optional_parameters);
      printf("Import '%s'.\n", new_node_string);
      wb_supervisor_field_import_mf_node_from_string(root_children_field, -1, new_node_string);
      printf("Added '%s' at position (%f, %f).\n", def_name, x, y);
    } else if (strncmp(message, "remove obstacle", strlen("remove obstacle")) == 0) {
      char def_name[64];
      const int found = sscanf(message, "remove obstacle %s", def_name);
      if (found != 1) {
        fprintf(stderr, "Invalid instructions to remove obstacle.\n");
        continue;
      }
      printf("DEF name %s: %s\n", def_name, message);
      WbNodeRef node = wb_supervisor_node_get_from_def(def_name);
      if (!node) {
        fprintf(stderr, "Invalid obstacle DEF name to be removed.\n");
        continue;
      }
      wb_supervisor_node_remove(node);
      printf("Removed obstacle '%s'\n", def_name);
    } else
      // This should not occur.
      fprintf(stderr, "Unkown message: '%s'\n", message);
  }
}

void wb_robot_window_cleanup() {
  // This is called when the robot window is destroyed.
  // There is nothing to do here in this example.
  // This callback can be used to store information.
}

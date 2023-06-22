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

#include "quaternion_private.h"
#include "vector3_private.h"
#include <stdio.h>
#include <stdlib.h>

WbuQuaternion wbu_quaternion_zero() {
  WbuQuaternion q;
  q.w = 1.0;
  q.x = 0.0;
  q.y = 0.0;
  q.z = 0.0;
  return q;
}

WbuQuaternion wbu_quaternion(double w, double x, double y, double z) {
  WbuQuaternion res;
  res.w = w;
  res.x = x;
  res.y = y;
  res.z = z;
  return res;
}

WbuQuaternion wbu_quaternion_normalize(WbuQuaternion q) {
  WbuQuaternion res;
  double n = sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n == 0) {
    fprintf(stderr, "Error: Trying to normalize a zero quaternion!\n");
    return q;
  }
  res.w = q.w / n;
  res.x = q.x / n;
  res.y = q.y / n;
  res.z = q.z / n;

  return res;
}

WbuQuaternion wbu_quaternion_multiply(WbuQuaternion x, WbuQuaternion y) {
  WbuQuaternion res;
  res.w = y.w * x.w - y.x * x.x - y.y * x.y - y.z * x.z;
  res.x = y.w * x.x + y.x * x.w - y.y * x.z + y.z * x.y;
  res.y = y.w * x.y + y.x * x.z + y.y * x.w - y.z * x.x;
  res.z = y.w * x.z - y.x * x.y + y.y * x.x + y.z * x.w;
  return res;
}

WbuQuaternion wbu_quaternion_conjugate(WbuQuaternion q) {
  WbuQuaternion res;
  res.w = q.w;
  res.x = -q.x;
  res.y = -q.y;
  res.z = -q.z;
  return wbu_quaternion_normalize(res);
}

WbuQuaternion wbu_quaternion_from_axis_angle(double x, double y, double z, double angle) {
  double half_angle = angle * 0.5;
  double sin_angle = sin(half_angle);
  WbuVector3 axis;
  axis.x = x;
  axis.y = y;
  axis.z = z;
  axis = wbu_vector3_normalize(axis);
  WbuQuaternion res;
  res.w = cos(half_angle);
  res.x = sin_angle * axis.x;
  res.y = sin_angle * axis.y;
  res.z = sin_angle * axis.z;
  res = wbu_quaternion_normalize(res);
  return res;
}

void wbu_quaternion_to_axis_angle(WbuQuaternion q, double *axis_angle) {
  // if q.w > 1, acos will return nan
  // if this actually happens we should normalize the quaternion here
  if (q.w > 1.0)
    wbu_quaternion_normalize(q);
  if (q.w < 1.0)
    axis_angle[3] = 2.0 * acos(q.w);
  else
    // q.w could still be slightly greater than 1.0 (floating point inaccuracy)
    axis_angle[3] = 0.0;

  if (axis_angle[3] < 0.0001) {
    // if e[3] close to zero then direction of axis not important
    axis_angle[0] = 0.0;
    axis_angle[1] = 1.0;
    axis_angle[2] = 0.0;
    axis_angle[3] = 0.0;
    return;
  }

  // normalize axes
  const double inv = 1.0 / sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  axis_angle[0] = q.x * inv;
  axis_angle[1] = q.y * inv;
  axis_angle[2] = q.z * inv;
}

void wbu_quaternion_print(WbuQuaternion q) {
  printf("quaternion %f %f %f %f\n", q.w, q.x, q.y, q.z);
}

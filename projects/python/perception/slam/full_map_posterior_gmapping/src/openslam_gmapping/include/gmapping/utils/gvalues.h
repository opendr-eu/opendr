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

#ifndef _GVALUES_H_
#define _GVALUES_H_

#ifdef LINUX
#include <values.h>
#endif
#ifdef MACOSX
#include <limits.h>
#include <math.h>
#define MAXDOUBLE 1e1000
//#define isnan(x) (x==FP_NAN)
#endif
#ifdef _WIN32
#include <limits>
#ifndef __DRAND48_DEFINED__
#define __DRAND48_DEFINED__
inline double drand48() {
  return double(rand()) / RAND_MAX;
}
#endif
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#define round(d) (floor((d) + 0.5))
typedef unsigned int uint;
#define isnan(x) (_isnan(x))
#endif

#endif

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
#ifndef C_API_TARGET_H
#define C_API_TARGET_H

#ifdef __cplusplus
extern "C" {
#endif

/***
 * OpenDR data type for representing classification targets
 */
struct OpendrCategoryTarget {
  int data;
  float confidence;
};
typedef struct OpendrCategoryTarget OpendrCategoryTargetT;

/***
 * OpenDR data type for representing detection targets
 */
struct OpendrDetectionTarget {
  int name;
  float left;
  float top;
  float width;
  float height;
  float score;
};
typedef struct OpendrDetectionTarget OpendrDetectionTargetT;

/***
 * OpenDR data type for representing vectors of detections targets
 */
struct OpendrDetectionVectorTarget {
  OpendrDetectionTargetT *startingPointer;
  int size;
};
typedef struct OpendrDetectionVectorTarget OpendrDetectionVectorTargetT;

#ifdef __cplusplus
}
#endif

#endif  // C_API_TARGET_H

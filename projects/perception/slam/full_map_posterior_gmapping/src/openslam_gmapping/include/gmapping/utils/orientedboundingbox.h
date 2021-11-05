
/*
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

#ifndef ORIENTENDBOUNDINGBOX_H
#define ORIENTENDBOUNDINGBOX_H

#include <math.h>
#include <stdio.h>

#include <gmapping/utils/point.h>

namespace GMapping {

  template<class NUMERIC> class OrientedBoundingBox {
  public:
    OrientedBoundingBox(std::vector<point<NUMERIC>> p);

    double area();

  protected:
    Point ul;
    Point ur;
    Point ll;
    Point lr;
  };

#include "gmapping/utils/orientedboundingbox.hxx"

};  // namespace GMapping

#endif

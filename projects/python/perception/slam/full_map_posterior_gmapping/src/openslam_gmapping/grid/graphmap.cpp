// Copyright 2020-2023 OpenDR European Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GRAPHMAP_H
#define GRAPHMAP_H

#include <gmapping/grid/map.h>
#include <gmapping/utils/point.h>
#include <utils/graph.h>
#include <list>

namespace GMapping {

  class RasterMap;

  struct GraphMapPatch {
    typedef typename std::list<IntPoint> PointList;
    /**Renders the map relatively to the center of the patch*/
    // void render(RenderMap rmap);
    /**returns the lower left corner of the patch, relative to the center*/
    // Point minBoundary() const;
    /**returns the upper right corner of the patch, relative to the center*/
    // Point maxBoundary() const; //

    OrientedPoint center;
    PointList m_points;
  };

  struct Covariance3 {
    double sxx, sxy, sxt, syy, syt, stt;
  };

  struct GraphMapEdge {
    Covariance3 covariance;
    GraphMapPatch *first, *second;

    inline operator double() const { return sqrt((first->center - second->center) * (first->center - second->center)); }
  };

  struct GraphPatchGraph : public Graph<GraphMapPatch, Covariance3> {
    void addEdge(Vertex *v1, Vertex *v2, const Covariance3 &covariance);
  };

  void GraphPatchGraph::addEdge(GraphPatchGraph::Vertex *v1, GraphPatchGraph::VertexVertex *v2, const Covariance3 &cov) {
    GraphMapEdge gme;
    gme.covariance = cov;
    gme.first = v1;
    gme.second = v2;
    return Graph<GraphMapPatch, Covariance3>::addEdge(v1, v2, gme);
  }

  struct GraphPatchDirectoryCell : public std::set<GraphMapPatch::Vertex *> {
    GraphPatchDirectoryCell(double);
  };

  // clang-format off
  // Ignore clang due to it going back-and-forth between "...Cell>>" and "...Cell> >"
  typedef Map<GraphPatchDirectoryCell>, Array2D::set<GraphPatchDirectoryCell>
  // clang-format on

};  // namespace GMapping

#endif

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

#ifndef MAP_H
#define MAP_H
#include <assert.h>
#include <gmapping/utils/point.h>
#include "gmapping/grid/accessstate.h"
#include "gmapping/grid/array2d.h"

namespace GMapping {
  /**
  The cells have to define the special value Cell::Unknown to handle with the unallocated areas.
  The cells have to define (int) constructor;
  */
  typedef Array2D<double> DoubleArray2D;

  template<class Cell, class Storage, const bool isClass = true> class Map {
  public:
    enum MapModel { ReflectionModel, ExpDecayModel };

    Map(int mapSizeX, int mapSizeY, double delta, MapModel mapModel = ReflectionModel);
    Map(const Point &center, double worldSizeX, double worldSizeY, double delta, MapModel mapModel = ReflectionModel);
    Map(const Point &center, double xmin, double ymin, double xmax, double ymax, double delta,
        MapModel mapModel = ReflectionModel);
    /* the standard implementation works filen in this case*/
    // Map(const Map& g);
    // Map& operator =(const Map& g);
    void resize(double xmin, double ymin, double xmax, double ymax);
    void grow(double xmin, double ymin, double xmax, double ymax);
    inline IntPoint world2map(const Point &p) const;
    inline Point map2world(const IntPoint &p) const;
    inline IntPoint world2map(double x, double y) const { return world2map(Point(x, y)); }
    inline Point map2world(int x, int y) const { return map2world(IntPoint(x, y)); }

    inline Point getCenter() const { return m_center; }
    inline double getWorldSizeX() const { return m_worldSizeX; }
    inline double getWorldSizeY() const { return m_worldSizeY; }
    inline int getMapSizeX() const { return m_mapSizeX; }
    inline int getMapSizeY() const { return m_mapSizeY; }
    inline double getDelta() const { return m_delta; }
    inline double getMapResolution() const { return m_delta; }
    inline double getResolution() const { return m_delta; }
    inline void getSize(double &xmin, double &ymin, double &xmax, double &ymax) const {
      Point min = map2world(0, 0), max = map2world(IntPoint(m_mapSizeX - 1, m_mapSizeY - 1));
      xmin = min.x, ymin = min.y, xmax = max.x, ymax = max.y;
    }

    inline Cell &cell(int x, int y) { return cell(IntPoint(x, y)); }
    inline Cell &cell(const IntPoint &p);

    inline const Cell &cell(int x, int y) const { return cell(IntPoint(x, y)); }
    inline const Cell &cell(const IntPoint &p) const;

    inline Cell &cell(double x, double y) { return cell(Point(x, y)); }
    inline Cell &cell(const Point &p);

    inline const Cell &cell(double x, double y) const { return cell(Point(x, y)); }

    inline double cell_value(const IntPoint &p);

    inline double cell_value(int x, int y) { return cell_value(IntPoint(x, y)); }

    inline double cell_value(const Point &p) { return cell_value(world2map(p)); }

    inline double cell_value(double x, double y) { return cell_value(Point(x, y)); }

    void setAlpha(double alpha) { m_alpha = alpha; }
    void setBeta(double beta) { m_beta = beta; }
    double getAlpha() const { return m_alpha; }
    double getBeta() const { return m_beta; }

    double alpha(const IntPoint &p);
    inline double alpha(int x, int y) const { return alpha(IntPoint(x, y)); }
    double beta(const IntPoint &p);
    inline double beta(int x, int y) const { return beta(IntPoint(x, y)); }

    inline bool isInside(int x, int y) const { return m_storage.cellState(IntPoint(x, y)) & Inside; }
    inline bool isInside(const IntPoint &p) const { return m_storage.cellState(p) & Inside; }

    inline bool isInside(double x, double y) const { return m_storage.cellState(world2map(x, y)) & Inside; }
    inline bool isInside(const Point &p) const { return m_storage.cellState(world2map(p)) & Inside; }

    inline const Cell &cell(const Point &p) const;

    inline Storage &storage() { return m_storage; }
    inline const Storage &storage() const { return m_storage; }
    DoubleArray2D *toDoubleArray() const;

    Map<double, DoubleArray2D, false> *toDoubleMap() const;

  protected:
    Point m_center;
    double m_worldSizeX, m_worldSizeY, m_delta;
    Storage m_storage;
    int m_mapSizeX, m_mapSizeY;
    int m_sizeX2, m_sizeY2;
    double m_alpha, m_beta;
    MapModel m_mapModel;
    static const Cell m_unknown;
  };

  typedef Map<double, DoubleArray2D, false> DoubleMap;

  template<class Cell, class Storage, const bool isClass> const Cell Map<Cell, Storage, isClass>::m_unknown = Cell(-1);

  template<class Cell, class Storage, const bool isClass>
  Map<Cell, Storage, isClass>::Map(int mapSizeX, int mapSizeY, double delta, MapModel mapModel) :
    m_storage(mapSizeX, mapSizeY) {
    m_worldSizeX = mapSizeX * delta;
    m_worldSizeY = mapSizeY * delta;
    m_delta = delta;
    m_center = Point(0.5 * m_worldSizeX, 0.5 * m_worldSizeY);
    m_sizeX2 = m_mapSizeX >> 1;
    m_sizeY2 = m_mapSizeY >> 1;

    m_mapModel = mapModel;
  }

  template<class Cell, class Storage, const bool isClass>
  Map<Cell, Storage, isClass>::Map(const Point &center, double worldSizeX, double worldSizeY, double delta, MapModel mapModel) :
    m_storage((int)ceil(worldSizeX / delta), (int)ceil(worldSizeY / delta)) {
    m_center = center;
    m_worldSizeX = worldSizeX;
    m_worldSizeY = worldSizeY;
    m_delta = delta;
    m_mapSizeX = m_storage.getXSize() << m_storage.getPatchSize();
    m_mapSizeY = m_storage.getYSize() << m_storage.getPatchSize();
    m_sizeX2 = m_mapSizeX >> 1;
    m_sizeY2 = m_mapSizeY >> 1;

    m_mapModel = mapModel;
  }

  template<class Cell, class Storage, const bool isClass>
  Map<Cell, Storage, isClass>::Map(const Point &center, double xmin, double ymin, double xmax, double ymax, double delta,
                                   MapModel mapModel) :
    m_storage((int)ceil((xmax - xmin) / delta), (int)ceil((ymax - ymin) / delta)) {
    m_center = center;
    m_worldSizeX = xmax - xmin;
    m_worldSizeY = ymax - ymin;
    m_delta = delta;
    m_mapSizeX = m_storage.getXSize() << m_storage.getPatchSize();
    m_mapSizeY = m_storage.getYSize() << m_storage.getPatchSize();
    m_sizeX2 = (int)round((m_center.x - xmin) / m_delta);
    m_sizeY2 = (int)round((m_center.y - ymin) / m_delta);

    m_mapModel = mapModel;
  }

  template<class Cell, class Storage, const bool isClass>
  void Map<Cell, Storage, isClass>::resize(double xmin, double ymin, double xmax, double ymax) {
    IntPoint imin = world2map(xmin, ymin);
    IntPoint imax = world2map(xmax, ymax);
    int pxmin, pymin, pxmax, pymax;
    pxmin = (int)floor((float)imin.x / (1 << m_storage.getPatchMagnitude()));
    pxmax = (int)ceil((float)imax.x / (1 << m_storage.getPatchMagnitude()));
    pymin = (int)floor((float)imin.y / (1 << m_storage.getPatchMagnitude()));
    pymax = (int)ceil((float)imax.y / (1 << m_storage.getPatchMagnitude()));
    m_storage.resize(pxmin, pymin, pxmax, pymax);
    m_mapSizeX = m_storage.getXSize() << m_storage.getPatchSize();
    m_mapSizeY = m_storage.getYSize() << m_storage.getPatchSize();
    m_worldSizeX = xmax - xmin;
    m_worldSizeY = ymax - ymin;
    m_sizeX2 -= pxmin * (1 << m_storage.getPatchMagnitude());
    m_sizeY2 -= pymin * (1 << m_storage.getPatchMagnitude());
  }

  template<class Cell, class Storage, const bool isClass>
  void Map<Cell, Storage, isClass>::grow(double xmin, double ymin, double xmax, double ymax) {
    IntPoint imin = world2map(xmin, ymin);
    IntPoint imax = world2map(xmax, ymax);
    if (isInside(imin) && isInside(imax))
      return;
    imin = min(imin, IntPoint(0, 0));
    imax = max(imax, IntPoint(m_mapSizeX - 1, m_mapSizeY - 1));
    int pxmin, pymin, pxmax, pymax;
    pxmin = (int)floor((float)imin.x / (1 << m_storage.getPatchMagnitude()));
    pxmax = (int)ceil((float)imax.x / (1 << m_storage.getPatchMagnitude()));
    pymin = (int)floor((float)imin.y / (1 << m_storage.getPatchMagnitude()));
    pymax = (int)ceil((float)imax.y / (1 << m_storage.getPatchMagnitude()));
    m_storage.resize(pxmin, pymin, pxmax, pymax);
    m_mapSizeX = m_storage.getXSize() << m_storage.getPatchSize();
    m_mapSizeY = m_storage.getYSize() << m_storage.getPatchSize();
    m_worldSizeX = xmax - xmin;
    m_worldSizeY = ymax - ymin;
    m_sizeX2 -= pxmin * (1 << m_storage.getPatchMagnitude());
    m_sizeY2 -= pymin * (1 << m_storage.getPatchMagnitude());
  }

  template<class Cell, class Storage, const bool isClass>
  IntPoint Map<Cell, Storage, isClass>::world2map(const Point &p) const {
    return IntPoint((int)round((p.x - m_center.x) / m_delta) + m_sizeX2, (int)round((p.y - m_center.y) / m_delta) + m_sizeY2);
  }

  template<class Cell, class Storage, const bool isClass>
  Point Map<Cell, Storage, isClass>::map2world(const IntPoint &p) const {
    return Point((p.x - m_sizeX2) * m_delta, (p.y - m_sizeY2) * m_delta) + m_center;
  }

  template<class Cell, class Storage, const bool isClass> Cell &Map<Cell, Storage, isClass>::cell(const IntPoint &p) {
    AccessibilityState s = m_storage.cellState(p);
    if (!s & Inside)
      assert(0);
    // if (s&Allocated) return m_storage.cell(p); assert(0);

    // this will never happend. Just to satify the compiler..
    return m_storage.cell(p);
  }

  template<class Cell, class Storage, const bool isClass> Cell &Map<Cell, Storage, isClass>::cell(const Point &p) {
    IntPoint ip = world2map(p);
    AccessibilityState s = m_storage.cellState(ip);
    if (!s & Inside)
      assert(0);
    // if (s&Allocated) return m_storage.cell(ip); assert(0);

    // this will never happend. Just to satify the compiler..
    return m_storage.cell(ip);
  }

  template<class Cell, class Storage, const bool isClass> double Map<Cell, Storage, isClass>::cell_value(const IntPoint &p) {
    AccessibilityState s = m_storage.cellState(p);
    if (!s & Inside)
      assert(0);

    Cell cell = m_storage.cell(p);

    if (m_mapModel == ReflectionModel)
      return (double)cell;

    if (m_mapModel == ExpDecayModel)
      return cell.R != 0 ? cell.n / cell.R : -1;
  }

  template<class Cell, class Storage, const bool isClass> double Map<Cell, Storage, isClass>::alpha(const IntPoint &p) {
    AccessibilityState s = m_storage.cellState(p);
    if (!s & Inside)
      assert(0);

    Cell cell = m_storage.cell(p);
    return cell.n;
  }

  template<class Cell, class Storage, const bool isClass> double Map<Cell, Storage, isClass>::beta(const IntPoint &p) {
    AccessibilityState s = m_storage.cellState(p);
    if (!s & Inside)
      assert(0);

    Cell cell = m_storage.cell(p);
    if (m_mapModel == ExpDecayModel)
      return cell.R;
    else
      return cell.visits - cell.n;
  }

  template<class Cell, class Storage, const bool isClass>
  const Cell &Map<Cell, Storage, isClass>::cell(const IntPoint &p) const {
    AccessibilityState s = m_storage.cellState(p);
    // if (! s&Inside) assert(0);
    if (s & Allocated)
      return m_storage.cell(p);
    return m_unknown;
  }

  template<class Cell, class Storage, const bool isClass> const Cell &Map<Cell, Storage, isClass>::cell(const Point &p) const {
    IntPoint ip = world2map(p);
    AccessibilityState s = m_storage.cellState(ip);
    // if (! s&Inside) assert(0);
    if (s & Allocated)
      return m_storage.cell(ip);
    return m_unknown;
  }

  // FIXME check why the last line of the map is corrupted.
  template<class Cell, class Storage, const bool isClass> DoubleArray2D *Map<Cell, Storage, isClass>::toDoubleArray() const {
    DoubleArray2D *darr = new DoubleArray2D(getMapSizeX() - 1, getMapSizeY() - 1);
    for (int x = 0; x < getMapSizeX() - 1; x++)
      for (int y = 0; y < getMapSizeY() - 1; y++) {
        IntPoint p(x, y);
        darr->cell(p) = cell(p);
      }
    return darr;
  }

  template<class Cell, class Storage, const bool isClass>
  Map<double, DoubleArray2D, false> *Map<Cell, Storage, isClass>::toDoubleMap() const {
    // FIXME size the map so that m_center will be setted accordingly
    Point pmin = map2world(IntPoint(0, 0));
    Point pmax = map2world(getMapSizeX() - 1, getMapSizeY() - 1);
    Point center = (pmax + pmin) * 0.5;
    Map<double, DoubleArray2D, false> *plainMap =
      new Map<double, DoubleArray2D, false>(center, (pmax - pmin).x, (pmax - pmin).y, getDelta());
    for (int x = 0; x < getMapSizeX() - 1; x++)
      for (int y = 0; y < getMapSizeY() - 1; y++) {
        IntPoint p(x, y);
        plainMap->cell(p) = cell(p);
      }
    return plainMap;
  }

};  // namespace GMapping

#endif

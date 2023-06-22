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

#include "gmapping/scanmatcher/scanmatcher.h"

#include <cstring>
#include <iostream>
#include <limits>
#include <list>
//#define GENERATE_MAPS

namespace GMapping {

  using namespace std;

  const double ScanMatcher::nullLikelihood = -.5;

  ScanMatcher::ScanMatcher() : m_laserPose(0, 0, 0) {
    // m_laserAngles=0;
    m_laserBeams = 0;
    m_optRecursiveIterations = 3;
    m_activeAreaComputed = false;

    // This  are the dafault settings for a grid map of 5 cm
    m_llsamplerange = 0.01;
    m_llsamplestep = 0.01;
    m_lasamplerange = 0.005;
    m_lasamplestep = 0.005;
    m_enlargeStep = 10.;
    m_fullnessThreshold = 0.1;
    m_angularOdometryReliability = 0.;
    m_linearOdometryReliability = 0.;
    m_freeCellRatio = sqrt(2.);
    m_initialBeamsSkip = 0;

    /*
            // This  are the dafault settings for a grid map of 10 cm
            m_llsamplerange=0.1;
            m_llsamplestep=0.1;
            m_lasamplerange=0.02;
            m_lasamplestep=0.01;
    */
    // This  are the dafault settings for a grid map of 20/25 cm
    /*
            m_llsamplerange=0.2;
            m_llsamplestep=0.1;
            m_lasamplerange=0.02;
            m_lasamplestep=0.01;
            m_generateMap=false;
    */

    m_linePoints = new IntPoint[60000];

    m_mapModel = ScanMatcherMap::MapModel::ReflectionModel;
    m_particleWeighting = ClosestMeanHitLikelihood;
  }

  ScanMatcher::~ScanMatcher() { delete[] m_linePoints; }

  void ScanMatcher::invalidateActiveArea() { m_activeAreaComputed = false; }

  /*
  void ScanMatcher::computeActiveArea(ScanMatcherMap& map, const OrientedPoint& p, const double* readings){
          if (m_activeAreaComputed)
                  return;
          HierarchicalArray2D<PointAccumulator>::PointSet activeArea;
          OrientedPoint lp=p;
          lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
          lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
          lp.theta+=m_laserPose.theta;
          IntPoint p0=map.world2map(lp);
          const double * angle=m_laserAngles;
          for (const double* r=readings; r<readings+m_laserBeams; r++, angle++)
                  if (m_generateMap){
                          double d=*r;
                          if (d>m_laserMaxRange)
                                  continue;
                          if (d>m_usableRange)
                                  d=m_usableRange;

                          Point phit=lp+Point(d*cos(lp.theta+*angle),d*sin(lp.theta+*angle));
                          IntPoint p1=map.world2map(phit);

                          d+=map.getDelta();
                          //Point phit2=lp+Point(d*cos(lp.theta+*angle),d*sin(lp.theta+*angle));
                          //IntPoint p2=map.world2map(phit2);
                          IntPoint linePoints[20000] ;
                          GridLineTraversalLine line;
                          line.points=linePoints;
                          //GridLineTraversal::gridLine(p0, p2, &line);
                          GridLineTraversal::gridLine(p0, p1, &line);
                          for (int i=0; i<line.num_points-1; i++){
                                  activeArea.insert(map.storage().patchIndexes(linePoints[i]));
                          }
                          if (d<=m_usableRange){
                                  activeArea.insert(map.storage().patchIndexes(p1));
                                  //activeArea.insert(map.storage().patchIndexes(p2));
                          }
                  } else {
                          if (*r>m_laserMaxRange||*r>m_usableRange) continue;
                          Point phit=lp;
                          phit.x+=*r*cos(lp.theta+*angle);
                          phit.y+=*r*sin(lp.theta+*angle);
                          IntPoint p1=map.world2map(phit);
                          assert(p1.x>=0 && p1.y>=0);
                          IntPoint cp=map.storage().patchIndexes(p1);
                          assert(cp.x>=0 && cp.y>=0);
                          activeArea.insert(cp);

                  }
          //this allocates the unallocated cells in the active area of the map
          //cout << "activeArea::size() " << activeArea.size() << endl;
          map.storage().setActiveArea(activeArea, true);
          m_activeAreaComputed=true;
  }
  */
  void ScanMatcher::computeActiveArea(ScanMatcherMap &map, const OrientedPoint &p, const double *readings) {
    if (m_activeAreaComputed)
      return;
    OrientedPoint lp = p;
    lp.x += cos(p.theta) * m_laserPose.x - sin(p.theta) * m_laserPose.y;
    lp.y += sin(p.theta) * m_laserPose.x + cos(p.theta) * m_laserPose.y;
    lp.theta += m_laserPose.theta;
    IntPoint p0 = map.world2map(lp);

    Point min(map.map2world(0, 0));
    Point max(map.map2world(map.getMapSizeX() - 1, map.getMapSizeY() - 1));

    if (lp.x < min.x)
      min.x = lp.x;
    if (lp.y < min.y)
      min.y = lp.y;
    if (lp.x > max.x)
      max.x = lp.x;
    if (lp.y > max.y)
      max.y = lp.y;

    /*determine the size of the area*/
    const double *angle = m_laserAngles + m_initialBeamsSkip;
    for (const double *r = readings + m_initialBeamsSkip; r < readings + m_laserBeams; r++, angle++) {
      if (*r > m_laserMaxRange || *r == 0.0 || isnan(*r))
        continue;
      double d = *r > m_usableRange ? m_usableRange : *r;
      Point phit = lp;
      phit.x += d * cos(lp.theta + *angle);
      phit.y += d * sin(lp.theta + *angle);
      if (phit.x < min.x)
        min.x = phit.x;
      if (phit.y < min.y)
        min.y = phit.y;
      if (phit.x > max.x)
        max.x = phit.x;
      if (phit.y > max.y)
        max.y = phit.y;
    }
    // min=min-Point(map.getDelta(),map.getDelta());
    // max=max+Point(map.getDelta(),map.getDelta());

    if (!map.isInside(min) || !map.isInside(max)) {
      Point lmin(map.map2world(0, 0));
      Point lmax(map.map2world(map.getMapSizeX() - 1, map.getMapSizeY() - 1));
      // cerr << "CURRENT MAP " << lmin.x << " " << lmin.y << " " << lmax.x << " " << lmax.y << endl;
      // cerr << "BOUNDARY OVERRIDE " << min.x << " " << min.y << " " << max.x << " " << max.y << endl;
      min.x = (min.x >= lmin.x) ? lmin.x : min.x - m_enlargeStep;
      max.x = (max.x <= lmax.x) ? lmax.x : max.x + m_enlargeStep;
      min.y = (min.y >= lmin.y) ? lmin.y : min.y - m_enlargeStep;
      max.y = (max.y <= lmax.y) ? lmax.y : max.y + m_enlargeStep;
      map.resize(min.x, min.y, max.x, max.y);
      // cerr << "RESIZE " << min.x << " " << min.y << " " << max.x << " " << max.y << endl;
    }

    HierarchicalArray2D<PointAccumulator>::PointSet activeArea;
    /*allocate the active area*/
    angle = m_laserAngles + m_initialBeamsSkip;
    for (const double *r = readings + m_initialBeamsSkip; r < readings + m_laserBeams; r++, angle++)
      if (m_generateMap) {
        double d = *r;
        if (d > m_laserMaxRange || d == 0.0 || isnan(d))
          continue;
        if (d > m_usableRange)
          d = m_usableRange;
        Point phit = lp + Point(d * cos(lp.theta + *angle), d * sin(lp.theta + *angle));
        IntPoint lp0 = map.world2map(lp);
        IntPoint lp1 = map.world2map(phit);

        // IntPoint linePoints[20000] ;
        GridLineTraversalLine line;
        line.points = m_linePoints;
        GridLineTraversal::gridLine(lp0, lp1, &line);
        for (int i = 0; i < line.num_points - 1; i++) {
          assert(map.isInside(m_linePoints[i]));
          activeArea.insert(map.storage().patchIndexes(m_linePoints[i]));
          assert(m_linePoints[i].x >= 0 && m_linePoints[i].y >= 0);
        }
        if (d < m_usableRange) {
          IntPoint cp = map.storage().patchIndexes(lp1);
          assert(cp.x >= 0 && cp.y >= 0);
          activeArea.insert(cp);
        }
      } else {
        if (*r > m_laserMaxRange || *r > m_usableRange || *r == 0.0 || isnan(*r))
          continue;
        Point phit = lp;
        phit.x += *r * cos(lp.theta + *angle);
        phit.y += *r * sin(lp.theta + *angle);
        IntPoint p1 = map.world2map(phit);
        assert(p1.x >= 0 && p1.y >= 0);
        IntPoint cp = map.storage().patchIndexes(p1);
        assert(cp.x >= 0 && cp.y >= 0);
        activeArea.insert(cp);
      }

    // this allocates the unallocated cells in the active area of the map
    // cout << "activeArea::size() " << activeArea.size() << endl;
    /*
            cerr << "ActiveArea=";
            for (HierarchicalArray2D<PointAccumulator>::PointSet::const_iterator it=activeArea.begin(); it!= activeArea.end();
       it++){ cerr << "(" << it->x <<"," << it->y << ") ";
            }
            cerr << endl;
    */
    map.storage().setActiveArea(activeArea, true);
    m_activeAreaComputed = true;
  }

  /**
   * Computes the length of the beam segment that crosses through a given map cell.
   * Used for the Decay Rate model.
   * # TODO : Check if GridLineTraversal uses Bresenham algorithm (does not include all cells crossed by beam).
   * # TODO : If so, try using Wu's algorithm with aliasing to compute r_i as well.
   *
   * @param map (const ScanMatcherMap) Map with discretized cells
   * @param beamStart (Point) The x,y position of the laser sensor
   * @param beamEnd (Point) The x,y position of the beam's endpoint (hit or maxrange)
   * @param cell (IntPoint) The x,y position of the cell to be checked in integer units (discrete)
   * @return (double) The length r_i of the segment of beam contained within the cell. Returns 0 if the beam is outside the
   * cell.
   */
  double ScanMatcher::computeCellR(const ScanMatcherMap &map, Point beamStart, Point beamEnd, IntPoint cell) const {
    // Only execute for Exponential Decay Map Models
    if (m_mapModel != ScanMatcherMap::MapModel::ExpDecayModel)
      return 0;

    bool cellIsStartPt = map.world2map(beamStart) == cell, cellIsEndPt = map.world2map(beamEnd) == cell;

    Point deltaB = beamEnd - beamStart;
    // If the beam starts and ends inside the cell
    if (cellIsStartPt && cellIsEndPt)
      return euclidianDist(beamStart, beamEnd);

    Point cellCenter = map.map2world(cell);
    double delta = map.getDelta() / 2;

    // Current Cell horizontal and vertical grid lines
    Point cb0 = cellCenter - Point(delta, delta), cb1 = cellCenter + Point(delta, delta);

    bool cx0InsideBeam = (beamStart.x <= cb0.x && cb0.x <= beamEnd.x) || (beamEnd.x <= cb0.x && cb0.x <= beamStart.x);
    bool cy0InsideBeam = (beamStart.y <= cb0.y && cb0.y <= beamEnd.y) || (beamEnd.y <= cb0.y && cb0.y <= beamStart.y);
    bool cx1InsideBeam = (beamStart.x <= cb1.x && cb1.x <= beamEnd.x) || (beamEnd.x <= cb1.x && cb1.x <= beamStart.x);
    bool cy1InsideBeam = (beamStart.y <= cb1.y && cb1.y <= beamEnd.y) || (beamEnd.y <= cb1.y && cb1.y <= beamStart.y);

    // If cell not inside of beam
    if (!(cx0InsideBeam || cx1InsideBeam || cy0InsideBeam || cy1InsideBeam))
      return 0;

    if (abs(deltaB.x) < 1e-6) {  // Beam is vertical
      if (cellIsStartPt) {
        double cy = cy0InsideBeam ? cb0.y : cb1.y;
        return abs(beamStart.y - cy);
      } else if (cellIsEndPt) {
        double cy = cy0InsideBeam ? cb0.y : cb1.y;
        return abs(beamEnd.y - cy);
      } else
        return 2 * delta;
    } else if (abs(deltaB.y) < 1e-6) {  // Beam is Horizontal
      if (cellIsStartPt) {
        double cx = cx0InsideBeam ? cb0.x : cb1.x;
        return abs(beamStart.x - cx);
      } else if (cellIsEndPt) {
        double cx = cx0InsideBeam ? cb0.x : cb1.x;
        return abs(beamEnd.x - cx);
      } else
        return 2 * delta;
    } else {
      double m = deltaB.y / deltaB.x, b = beamEnd.y - m * beamEnd.x;

      // Intersections of beam with grid lines
      double ix0 = (cb0.y - b) / m, ix1 = (cb1.y - b) / m, iy0 = m * cb0.x + b, iy1 = m * cb1.x + b, tmp;

      // Straighten up the interval order
      if (ix0 > ix1) {
        tmp = ix0;
        ix0 = ix1;
        ix1 = tmp;
      }
      if (iy0 > iy1) {
        tmp = iy0;
        iy0 = iy1;
        iy1 = tmp;
      }

      // If the intersections don't fall inside the cell
      // (Why shouldn't it though? Line algorithm might give cells that are not touched by the beam [Bresenham])
      if (cb0.x >= ix1 || ix0 >= cb1.x || cb0.y >= iy1 || iy0 >= cb1.y)
        return 0;

      // Find the intersections of the x and y intervals
      double ex0 = cb0.x > ix0 ? cb0.x : ix0, ey0 = cb0.y > iy0 ? cb0.y : iy0, ex1 = cb1.x > ix1 ? ix1 : cb1.x,
             ey1 = cb1.y > iy1 ? iy1 : cb1.y;

      Point start, end;

      // If the beam endpoints are in cell
      if (cellIsStartPt) {
        start = beamStart;

        /* Then, figure out which of the two intersection points lies within the beam
        to compute the distance between the beam start|end and the intersect point. */
        if (((beamStart.x <= ex0 && ex0 <= beamEnd.x) || (beamEnd.x <= ex0 && ex0 <= beamStart.x)) &&
            ((beamStart.y <= ey0 && ey0 <= beamEnd.y) || (beamEnd.y <= ey0 && ey0 <= beamStart.y))) {
          end.x = ex0;
          end.y = ey0;
        } else {
          end.x = ex1;
          end.y = ey1;
        }

      } else if (cellIsEndPt) {
        start = beamEnd;
        if (((beamStart.x <= ex0 && ex0 <= beamEnd.x) || (beamEnd.x <= ex0 && ex0 <= beamStart.x)) &&
            ((beamStart.y <= ey0 && ey0 <= beamEnd.y) || (beamEnd.y <= ey0 && ey0 <= beamStart.y))) {
          end.x = ex0;
          end.y = ey0;
        } else {
          end.x = ex1;
          end.y = ey1;
        }
      } else {
        start = Point(ex0, ey0);
        end = Point(ex1, ey1);
      }

      return euclidianDist(start, end);
    }

    return 0;
  }

  double ScanMatcher::registerScan(ScanMatcherMap &map, const OrientedPoint &p, const double *readings) {
    if (!m_activeAreaComputed)
      computeActiveArea(map, p, readings);

    // this operation replicates the cells that will be changed in the registration operation
    map.storage().allocActiveArea();

    OrientedPoint lp = p;
    lp.x += cos(p.theta) * m_laserPose.x - sin(p.theta) * m_laserPose.y;
    lp.y += sin(p.theta) * m_laserPose.x + cos(p.theta) * m_laserPose.y;
    lp.theta += m_laserPose.theta;
    IntPoint p0 = map.world2map(lp);

    const double *angle = m_laserAngles + m_initialBeamsSkip;
    double esum = 0;
    for (const double *r = readings + m_initialBeamsSkip; r < readings + m_laserBeams; r++, angle++)
      if (m_generateMap) {
        double d = *r;
        bool out_of_range = false;
        if (d >= m_laserMaxRange || d == 0.0 || isnan(d))
          continue;
        if (d >= m_usableRange) {
          out_of_range = true;
          d = m_usableRange;
        }
        Point phit = lp + Point(d * cos(lp.theta + *angle), d * sin(lp.theta + *angle));
        IntPoint p1 = map.world2map(phit);
        // IntPoint linePoints[20000] ;
        GridLineTraversalLine line;
        line.points = m_linePoints;
        GridLineTraversal::gridLine(p0, p1, &line);
        int i = 0;
        for (i = 0; i < line.num_points; i++) {
          PointAccumulator &cell = map.cell(line.points[i]);
          double e = -cell.entropy();
          // Cell was a hit if it is the last of the beam and it wasn't a max-range
          bool hit = !out_of_range && i == line.num_points - 1;
          // Hit point for point accumulation (only assigned when cell is hit)
          Point hit_point = hit ? phit : Point(0, 0);
          // Distance that the ray traveled inside the map cell
          double ri = computeCellR(map, lp, phit, line.points[i]);
          cell.update(hit, hit_point, ri);

          e += cell.entropy();
          esum += e;
        }
      } else {
        if (*r > m_laserMaxRange || *r > m_usableRange || *r == 0.0 || isnan(*r))
          continue;

        Point phit = lp;
        phit.x += *r * cos(lp.theta + *angle);
        phit.y += *r * sin(lp.theta + *angle);
        IntPoint p1 = map.world2map(phit);
        assert(p1.x >= 0 && p1.y >= 0);
        double ri = computeCellR(map, lp, phit, p1);
        map.cell(p1).update(true, phit, ri);
      }

    // cout  << "informationGain=" << -esum << endl;
    return esum;
  }

  /*
  void ScanMatcher::registerScan(ScanMatcherMap& map, const OrientedPoint& p, const double* readings){
          if (!m_activeAreaComputed)
                  computeActiveArea(map, p, readings);

          //this operation replicates the cells that will be changed in the registration operation
          map.storage().allocActiveArea();

          OrientedPoint lp=p;
          lp.x+=cos(p.theta)*m_laserPose.x-sin(p.theta)*m_laserPose.y;
          lp.y+=sin(p.theta)*m_laserPose.x+cos(p.theta)*m_laserPose.y;
          lp.theta+=m_laserPose.theta;
          IntPoint p0=map.world2map(lp);
          const double * angle=m_laserAngles;
          for (const double* r=readings; r<readings+m_laserBeams; r++, angle++)
                  if (m_generateMap){
                          double d=*r;
                          if (d>m_laserMaxRange)
                                  continue;
                          if (d>m_usableRange)
                                  d=m_usableRange;
                          Point phit=lp+Point(d*cos(lp.theta+*angle),d*sin(lp.theta+*angle));
                          IntPoint p1=map.world2map(phit);

                          IntPoint linePoints[20000] ;
                          GridLineTraversalLine line;
                          line.points=linePoints;
                          GridLineTraversal::gridLine(p0, p1, &line);
                          for (int i=0; i<line.num_points-1; i++){
                                  IntPoint ci=map.storage().patchIndexes(line.points[i]);
                                  if (map.storage().getActiveArea().find(ci)==map.storage().getActiveArea().end())
                                          cerr << "BIG ERROR" <<endl;
                                  map.cell(line.points[i]).update(false, Point(0,0));
                          }
                          if (d<=m_usableRange){

                                  map.cell(p1).update(true,phit);
                          }
                  } else {
                          if (*r>m_laserMaxRange||*r>m_usableRange) continue;
                          Point phit=lp;
                          phit.x+=*r*cos(lp.theta+*angle);
                          phit.y+=*r*sin(lp.theta+*angle);
                          map.cell(phit).update(true,phit);
                  }
  }

  */

  double ScanMatcher::icpOptimize(OrientedPoint &pnew, const ScanMatcherMap &map, const OrientedPoint &init,
                                  const double *readings) const {
    double currentScore;
    double sc = score(map, init, readings);
    OrientedPoint start = init;
    pnew = init;
    int iterations = 0;
    do {
      currentScore = sc;
      sc = icpStep(pnew, map, start, readings);
      // cerr << "pstart=" << start.x << " " <<start.y << " " << start.theta << endl;
      // cerr << "pret=" << pnew.x << " " <<pnew.y << " " << pnew.theta << endl;
      start = pnew;
      iterations++;
    } while (sc > currentScore);
    cerr << "i=" << iterations << endl;
    return currentScore;
  }

  double ScanMatcher::optimize(OrientedPoint &pnew, const ScanMatcherMap &map, const OrientedPoint &init,
                               const double *readings) const {
    double bestScore = -1;
    OrientedPoint currentPose = init;
    double currentScore = score(map, currentPose, readings);
    double adelta = m_optAngularDelta, ldelta = m_optLinearDelta;
    unsigned int refinement = 0;
    enum Move { Front, Back, Left, Right, TurnLeft, TurnRight, Done };
    /*	cout << __PRETTY_FUNCTION__<<  " readings: ";
            for (int i=0; i<m_laserBeams; i++){
                    cout << readings[i] << " ";
            }
            cout << endl;
    */
    int c_iterations = 0;
    do {
      if (bestScore >= currentScore) {
        refinement++;
        adelta *= .5;
        ldelta *= .5;
      }
      bestScore = currentScore;
      //		cout <<"score="<< currentScore << " refinement=" << refinement;
      //		cout <<  "pose=" << currentPose.x  << " " << currentPose.y << " " << currentPose.theta << endl;
      OrientedPoint bestLocalPose = currentPose;
      OrientedPoint localPose = currentPose;

      Move move = Front;
      do {
        localPose = currentPose;
        switch (move) {
          case Front:
            localPose.x += ldelta;
            move = Back;
            break;
          case Back:
            localPose.x -= ldelta;
            move = Left;
            break;
          case Left:
            localPose.y -= ldelta;
            move = Right;
            break;
          case Right:
            localPose.y += ldelta;
            move = TurnLeft;
            break;
          case TurnLeft:
            localPose.theta += adelta;
            move = TurnRight;
            break;
          case TurnRight:
            localPose.theta -= adelta;
            move = Done;
            break;
          default:;
        }

        double odo_gain = 1;
        if (m_angularOdometryReliability > 0.) {
          double dth = init.theta - localPose.theta;
          dth = atan2(sin(dth), cos(dth));
          dth *= dth;
          odo_gain *= exp(-m_angularOdometryReliability * dth);
        }
        if (m_linearOdometryReliability > 0.) {
          double dx = init.x - localPose.x;
          double dy = init.y - localPose.y;
          double drho = dx * dx + dy * dy;
          odo_gain *= exp(-m_linearOdometryReliability * drho);
        }
        double localScore = odo_gain * score(map, localPose, readings);

        if (localScore > currentScore) {
          currentScore = localScore;
          bestLocalPose = localPose;
        }
        c_iterations++;
      } while (move != Done);
      currentPose = bestLocalPose;
      //		cout << "currentScore=" << currentScore<< endl;
      // here we look for the best move;
    } while (currentScore > bestScore || refinement < m_optRecursiveIterations);
    // cout << __PRETTY_FUNCTION__ << "bestScore=" << bestScore<< endl;
    // cout << __PRETTY_FUNCTION__ << "iterations=" << c_iterations<< endl;
    pnew = currentPose;
    return bestScore;
  }

  struct ScoredMove {
    OrientedPoint pose;
    double score;
    double likelihood;
  };

  typedef std::list<ScoredMove> ScoredMoveList;

  double ScanMatcher::optimize(OrientedPoint &_mean, ScanMatcher::CovarianceMatrix &_cov, const ScanMatcherMap &map,
                               const OrientedPoint &init, const double *readings) const {
    ScoredMoveList moveList;
    double bestScore = -1;
    OrientedPoint currentPose = init;
    ScoredMove sm = {currentPose, 0, 0};
    unsigned int matched = likelihoodAndScore(sm.score, sm.likelihood, map, currentPose, readings);
    double currentScore = sm.score;
    moveList.push_back(sm);
    double adelta = m_optAngularDelta, ldelta = m_optLinearDelta;
    unsigned int refinement = 0;
    int count = 0;
    enum Move { Front, Back, Left, Right, TurnLeft, TurnRight, Done };
    do {
      if (bestScore >= currentScore) {
        refinement++;
        adelta *= .5;
        ldelta *= .5;
      }
      bestScore = currentScore;
      //		cout <<"score="<< currentScore << " refinement=" << refinement;
      //		cout <<  "pose=" << currentPose.x  << " " << currentPose.y << " " << currentPose.theta << endl;
      OrientedPoint bestLocalPose = currentPose;
      OrientedPoint localPose = currentPose;

      Move move = Front;
      do {
        localPose = currentPose;
        switch (move) {
          case Front:
            localPose.x += ldelta;
            move = Back;
            break;
          case Back:
            localPose.x -= ldelta;
            move = Left;
            break;
          case Left:
            localPose.y -= ldelta;
            move = Right;
            break;
          case Right:
            localPose.y += ldelta;
            move = TurnLeft;
            break;
          case TurnLeft:
            localPose.theta += adelta;
            move = TurnRight;
            break;
          case TurnRight:
            localPose.theta -= adelta;
            move = Done;
            break;
          default:;
        }
        double localScore, localLikelihood;

        double odo_gain = 1;
        if (m_angularOdometryReliability > 0.) {
          double dth = init.theta - localPose.theta;
          dth = atan2(sin(dth), cos(dth));
          dth *= dth;
          odo_gain *= exp(-m_angularOdometryReliability * dth);
        }
        if (m_linearOdometryReliability > 0.) {
          double dx = init.x - localPose.x;
          double dy = init.y - localPose.y;
          double drho = dx * dx + dy * dy;
          odo_gain *= exp(-m_linearOdometryReliability * drho);
        }
        localScore = odo_gain * score(map, localPose, readings);
        // update the score
        count++;
        matched = likelihoodAndScore(localScore, localLikelihood, map, localPose, readings);
        if (localScore > currentScore) {
          currentScore = localScore;
          bestLocalPose = localPose;
        }
        sm.score = localScore;
        sm.likelihood = localLikelihood;  //+log(odo_gain);
        sm.pose = localPose;
        moveList.push_back(sm);
        // update the move list
      } while (move != Done);
      currentPose = bestLocalPose;
      // cout << __PRETTY_FUNCTION__ << "currentScore=" << currentScore<< endl;
      // here we look for the best move;
    } while (currentScore > bestScore || refinement < m_optRecursiveIterations);
    // cout << __PRETTY_FUNCTION__ << "bestScore=" << bestScore<< endl;
    // cout << __PRETTY_FUNCTION__ << "iterations=" << count<< endl;

    // normalize the likelihood
    double lmin = 1e9;
    double lmax = -1e9;
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      lmin = it->likelihood < lmin ? it->likelihood : lmin;
      lmax = it->likelihood > lmax ? it->likelihood : lmax;
    }
    // cout << "lmin=" << lmin << " lmax=" << lmax<< endl;
    for (ScoredMoveList::iterator it = moveList.begin(); it != moveList.end(); it++) {
      it->likelihood = exp(it->likelihood - lmax);
      // cout << "l=" << it->likelihood << endl;
    }
    // compute the mean
    OrientedPoint mean(0, 0, 0);
    double lacc = 0;
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      mean = mean + it->pose * it->likelihood;
      lacc += it->likelihood;
    }
    mean = mean * (1. / lacc);
    // OrientedPoint delta=mean-currentPose;
    // cout << "delta.x=" << delta.x << " delta.y=" << delta.y << " delta.theta=" << delta.theta << endl;
    CovarianceMatrix cov = {0., 0., 0., 0., 0., 0.};
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      OrientedPoint delta = it->pose - mean;
      delta.theta = atan2(sin(delta.theta), cos(delta.theta));
      cov.xx += delta.x * delta.x * it->likelihood;
      cov.yy += delta.y * delta.y * it->likelihood;
      cov.tt += delta.theta * delta.theta * it->likelihood;
      cov.xy += delta.x * delta.y * it->likelihood;
      cov.xt += delta.x * delta.theta * it->likelihood;
      cov.yt += delta.y * delta.theta * it->likelihood;
    }
    cov.xx /= lacc, cov.xy /= lacc, cov.xt /= lacc, cov.yy /= lacc, cov.yt /= lacc, cov.tt /= lacc;

    _mean = currentPose;
    _cov = cov;
    return bestScore;
  }

  void ScanMatcher::setLaserParameters(unsigned int beams, double *angles, const OrientedPoint &lpose) {
    /*if (m_laserAngles)
        delete [] m_laserAngles;
    */
    assert(beams < LASER_MAXBEAMS);
    m_laserPose = lpose;
    m_laserBeams = beams;
    // m_laserAngles=new double[beams];
    memcpy(m_laserAngles, angles, sizeof(double) * m_laserBeams);
  }

  double ScanMatcher::likelihood(double &_lmax, OrientedPoint &_mean, CovarianceMatrix &_cov, const ScanMatcherMap &map,
                                 const OrientedPoint &p, const double *readings) {
    ScoredMoveList moveList;

    for (double xx = -m_llsamplerange; xx <= m_llsamplerange; xx += m_llsamplestep)
      for (double yy = -m_llsamplerange; yy <= m_llsamplerange; yy += m_llsamplestep)
        for (double tt = -m_lasamplerange; tt <= m_lasamplerange; tt += m_lasamplestep) {
          OrientedPoint rp = p;
          rp.x += xx;
          rp.y += yy;
          rp.theta += tt;

          ScoredMove sm;
          sm.pose = rp;

          likelihoodAndScore(sm.score, sm.likelihood, map, rp, readings);
          moveList.push_back(sm);
        }

    // OrientedPoint delta=mean-currentPose;
    // cout << "delta.x=" << delta.x << " delta.y=" << delta.y << " delta.theta=" << delta.theta << endl;
    // normalize the likelihood
    double lmax = -1e9;
    double lcum = 0;
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      lmax = it->likelihood > lmax ? it->likelihood : lmax;
    }
    for (ScoredMoveList::iterator it = moveList.begin(); it != moveList.end(); it++) {
      // it->likelihood=exp(it->likelihood-lmax);
      lcum += exp(it->likelihood - lmax);
      it->likelihood = exp(it->likelihood - lmax);
      // cout << "l=" << it->likelihood << endl;
    }

    OrientedPoint mean(0, 0, 0);
    double s = 0, c = 0;
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      mean = mean + it->pose * it->likelihood;
      s += it->likelihood * sin(it->pose.theta);
      c += it->likelihood * cos(it->pose.theta);
    }
    mean = mean * (1. / lcum);
    s /= lcum;
    c /= lcum;
    mean.theta = atan2(s, c);

    CovarianceMatrix cov = {0., 0., 0., 0., 0., 0.};
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      OrientedPoint delta = it->pose - mean;
      delta.theta = atan2(sin(delta.theta), cos(delta.theta));
      cov.xx += delta.x * delta.x * it->likelihood;
      cov.yy += delta.y * delta.y * it->likelihood;
      cov.tt += delta.theta * delta.theta * it->likelihood;
      cov.xy += delta.x * delta.y * it->likelihood;
      cov.xt += delta.x * delta.theta * it->likelihood;
      cov.yt += delta.y * delta.theta * it->likelihood;
    }
    cov.xx /= lcum, cov.xy /= lcum, cov.xt /= lcum, cov.yy /= lcum, cov.yt /= lcum, cov.tt /= lcum;

    _mean = mean;
    _cov = cov;
    _lmax = lmax;
    return log(lcum) + lmax;
  }

  double ScanMatcher::likelihood(double &_lmax, OrientedPoint &_mean, CovarianceMatrix &_cov, const ScanMatcherMap &map,
                                 const OrientedPoint &p, Gaussian3 &odometry, const double *readings, double gain) {
    ScoredMoveList moveList;

    for (double xx = -m_llsamplerange; xx <= m_llsamplerange; xx += m_llsamplestep)
      for (double yy = -m_llsamplerange; yy <= m_llsamplerange; yy += m_llsamplestep)
        for (double tt = -m_lasamplerange; tt <= m_lasamplerange; tt += m_lasamplestep) {
          OrientedPoint rp = p;
          rp.x += xx;
          rp.y += yy;
          rp.theta += tt;

          ScoredMove sm;
          sm.pose = rp;

          likelihoodAndScore(sm.score, sm.likelihood, map, rp, readings);
          sm.likelihood += odometry.eval(rp) / gain;
          assert(!isnan(sm.likelihood));
          moveList.push_back(sm);
        }

    // OrientedPoint delta=mean-currentPose;
    // cout << "delta.x=" << delta.x << " delta.y=" << delta.y << " delta.theta=" << delta.theta << endl;
    // normalize the likelihood
    double lmax = -std::numeric_limits<double>::max();
    double lcum = 0;
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      lmax = it->likelihood > lmax ? it->likelihood : lmax;
    }
    for (ScoredMoveList::iterator it = moveList.begin(); it != moveList.end(); it++) {
      // it->likelihood=exp(it->likelihood-lmax);
      lcum += exp(it->likelihood - lmax);
      it->likelihood = exp(it->likelihood - lmax);
      // cout << "l=" << it->likelihood << endl;
    }

    OrientedPoint mean(0, 0, 0);
    double s = 0, c = 0;
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      mean = mean + it->pose * it->likelihood;
      s += it->likelihood * sin(it->pose.theta);
      c += it->likelihood * cos(it->pose.theta);
    }
    mean = mean * (1. / lcum);
    s /= lcum;
    c /= lcum;
    mean.theta = atan2(s, c);

    CovarianceMatrix cov = {0., 0., 0., 0., 0., 0.};
    for (ScoredMoveList::const_iterator it = moveList.begin(); it != moveList.end(); it++) {
      OrientedPoint delta = it->pose - mean;
      delta.theta = atan2(sin(delta.theta), cos(delta.theta));
      cov.xx += delta.x * delta.x * it->likelihood;
      cov.yy += delta.y * delta.y * it->likelihood;
      cov.tt += delta.theta * delta.theta * it->likelihood;
      cov.xy += delta.x * delta.y * it->likelihood;
      cov.xt += delta.x * delta.theta * it->likelihood;
      cov.yt += delta.y * delta.theta * it->likelihood;
    }
    cov.xx /= lcum, cov.xy /= lcum, cov.xt /= lcum, cov.yy /= lcum, cov.yt /= lcum, cov.tt /= lcum;

    _mean = mean;
    _cov = cov;
    _lmax = lmax;
    double v = log(lcum) + lmax;
    assert(!isnan(v));
    return v;
  }

  void ScanMatcher::setMatchingParameters(double urange, double range, double sigma, int kernsize, double lopt, double aopt,
                                          int iterations, double likelihoodSigma, unsigned int likelihoodSkip,
                                          ScanMatcherMap::MapModel mapModel, ParticleWeighting particleWeighting,
                                          double overconfidenceUniformWeight) {
    m_usableRange = urange;
    m_laserMaxRange = range;
    m_kernelSize = kernsize;
    m_optLinearDelta = lopt;
    m_optAngularDelta = aopt;
    m_optRecursiveIterations = iterations;
    m_gaussianSigma = sigma;
    m_likelihoodSigma = likelihoodSigma;
    m_likelihoodSkip = likelihoodSkip;

    m_mapModel = mapModel;
    m_particleWeighting = particleWeighting;
    m_overconfidenceUniformWeight = overconfidenceUniformWeight;
  }

  double ScanMatcher::closestMeanHitLikelihood(OrientedPoint &laser_pose, Point &end_point, double reading_range,
                                               double reading_bearing, const ScanMatcherMap &map, double &s,
                                               unsigned int &c) const {
    /*
     * Original Particle weight computation method from openslam_gmapping.
     * It computes a point pfree which is (roughly) one grid cell before the hit point in the
     * direction of the beam.
     * Then, for a given window around the hit point (+/- kernelSize in x and y), it checks if each cell
     * is occupied, and the cell in pfree relative to it is free according to a threshold value.
     * If it is the case, then the difference between the actual hitpoint and the mean of all hits of said cell
     * is computed. Finally, the minimum of all distances from hits and means is used to compute a
     * log likelihood from a gaussian, and added to the particle as it's weight.
     */

    if (reading_range > m_usableRange)
      return 0;

    double freeDelta = map.getDelta() * m_freeCellRatio;
    double noHit = nullLikelihood / (m_likelihoodSigma);

    IntPoint iphit = map.world2map(end_point);

    Point pfree = laser_pose;
    pfree.x += (reading_range - freeDelta) * cos(laser_pose.theta + reading_bearing);
    pfree.y += (reading_range - freeDelta) * sin(laser_pose.theta + reading_bearing);
    pfree = pfree - end_point;
    IntPoint ipfree = map.world2map(pfree);

    bool found = false;
    Point bestMu(0., 0.);
    for (int xx = -m_kernelSize; xx <= m_kernelSize; xx++)
      for (int yy = -m_kernelSize; yy <= m_kernelSize; yy++) {
        IntPoint pr = iphit + IntPoint(xx, yy);
        IntPoint pf = pr + ipfree;
        // AccessibilityState s=map.storage().cellState(pr);
        // if (s&Inside && s&Allocated){
        const PointAccumulator &cell = map.cell(pr);
        const PointAccumulator &fcell = map.cell(pf);
        if (((double)cell) > m_fullnessThreshold && ((double)fcell) < m_fullnessThreshold) {
          Point mu = end_point - cell.mean();
          if (!found) {
            bestMu = mu;
            found = true;
          } else
            bestMu = (mu * mu) < (bestMu * bestMu) ? mu : bestMu;
        }
        //}
      }
    if (found) {
      s += exp(-1. / m_gaussianSigma * bestMu * bestMu);
      c++;
    }
    double f = (-1. / m_likelihoodSigma) * (bestMu * bestMu);

    return (found) ? f : noHit;
  }

  double ScanMatcher::measurementLikelihood(OrientedPoint &laser_pose, Point &end_point, double reading_range,
                                            double reading_bearing, const ScanMatcherMap &map, double &s,
                                            unsigned int &c) const {
    double log_l = 0;

    IntPoint ilp = map.world2map(laser_pose);
    IntPoint iphit = map.world2map(end_point);

    bool out_of_range = reading_range >= m_usableRange;

    if (out_of_range)
      reading_range = m_usableRange;

    GridLineTraversalLine line;
    line.points = m_linePoints;
    GridLineTraversal::gridLine(ilp, iphit, &line);

    double alpha_prior = map.getAlpha();
    double beta_prior = map.getBeta();

    double numerator;
    double denominator;

    // For all the cells that the beam travelled through (misses)
    for (int i = 0; i < line.num_points; i++) {
      const IntPoint visited_cell_index = line.points[i];
      const PointAccumulator &visited_cell = map.cell(visited_cell_index);
      int Hi = visited_cell.n;
      // cell i reflected the beam if true, travelled through if false.
      bool delta_i = (i == line.num_points - 1) && !out_of_range;

      double l;

      if (m_mapModel == ScanMatcherMap::MapModel::ReflectionModel) {
        int Mi = visited_cell.visits - Hi;

        denominator = Hi + alpha_prior + Mi + beta_prior;

        // Ignore a cell if it hasn't been visited in the past.
        if (denominator == 0.0)
          continue;

        // If cell is the endpoint and was a hit
        if (delta_i)
          numerator = Hi + alpha_prior;
        // else, the beam travelled through it (or was max_range)
        else
          numerator = Mi + beta_prior;

        l = numerator / denominator;
        l = overconfidenceUniformNoise(l, out_of_range);

        if (l == 0)
          return std::numeric_limits<double>::quiet_NaN();
        else
          log_l += log(l);
      } else if (m_mapModel == ScanMatcherMap::MapModel::ExpDecayModel) {
        double ri = computeCellR(map, laser_pose, end_point, visited_cell_index);
        double Ri = visited_cell.R;
        double exponent = Hi + alpha_prior;

        numerator = Ri + beta_prior;
        denominator = numerator + ri;

        // Ignore a cell if it hasn't been visited in the past and,
        // somehow, neither this time (line discretization function returning cells not traversed by beam)
        // I.e.: both Ri and ri are 0
        if (denominator == 0.0)
          continue;

        // If cell is the endpoint and was a hit (not a max_range reading)
        if (delta_i)
          l = pow(numerator / denominator, exponent) * (exponent / denominator);
        // else, the beam travelled through it (or was a max_range)
        else
          l = pow(numerator / denominator, exponent);

        l = overconfidenceUniformNoise(l, out_of_range);

        if (l == 0)
          return std::numeric_limits<double>::quiet_NaN();
        else
          log_l += log(l);
      }
    }

    PointAccumulator cell = map.cell(iphit);
    if (!out_of_range && cell.n) {
      c++;
    }

    return log_l;
  }

  double ScanMatcher::overconfidenceUniformNoise(double l, bool out_of_range) const {
    // To avoid overconfidence, add random uniform noise.

    if (m_overconfidenceUniformWeight == 0)
      return l;

    double c2 = m_overconfidenceUniformWeight, c1 = 1 - c2;

    if (out_of_range)
      return (c1 * l) + c2;
    else
      return (c1 * l) + (c2 / m_usableRange);
  }

}  // namespace GMapping

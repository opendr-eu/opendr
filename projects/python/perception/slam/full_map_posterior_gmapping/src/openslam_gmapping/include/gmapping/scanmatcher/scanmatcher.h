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

#ifndef SCANMATCHER_H
#define SCANMATCHER_H

#include <gmapping/utils/gvalues.h>
#include <gmapping/utils/macro_params.h>
#include <gmapping/utils/stat.h>
#include <iostream>
#include <limits>
#include "gmapping/scanmatcher/gridlinetraversal.h"
#include "gmapping/scanmatcher/icp.h"
#include "gmapping/scanmatcher/smmap.h"

#define LASER_MAXBEAMS 2048

namespace GMapping {

  /*
  enum MapModel {
      ReflectionModel,
      ExpDecayModel
  };

  enum ParticleWeighting {
      ClosestMeanHitLikelihood,
      ForwardSensorModel,
      MeasurementLikelihood
  };
  */

  class ScanMatcher {
  public:
    typedef Covariance3 CovarianceMatrix;

    ScanMatcher();

    ~ScanMatcher();

    double icpOptimize(OrientedPoint &pnew, const ScanMatcherMap &map, const OrientedPoint &p, const double *readings) const;

    double optimize(OrientedPoint &pnew, const ScanMatcherMap &map, const OrientedPoint &p, const double *readings) const;

    double optimize(OrientedPoint &mean, CovarianceMatrix &cov, const ScanMatcherMap &map, const OrientedPoint &p,
                    const double *readings) const;

    double registerScan(ScanMatcherMap &map, const OrientedPoint &p, const double *readings);

    void setLaserParameters(unsigned int beams, double *angles, const OrientedPoint &lpose);

    enum ParticleWeighting { ClosestMeanHitLikelihood, ForwardSensorModel, MeasurementLikelihood };

    void setMatchingParameters(double urange, double range, double sigma, int kernsize, double lopt, double aopt,
                               int iterations, double likelihoodSigma = 1, unsigned int likelihoodSkip = 0,
                               ScanMatcherMap::MapModel mapModel = ScanMatcherMap::MapModel::ReflectionModel,
                               ParticleWeighting particleWeighting = ParticleWeighting::ClosestMeanHitLikelihood,
                               double overconfidenceUniformWeight = 0.05);

    void invalidateActiveArea();

    void computeActiveArea(ScanMatcherMap &map, const OrientedPoint &p, const double *readings);

    inline double icpStep(OrientedPoint &pret, const ScanMatcherMap &map, const OrientedPoint &p, const double *readings) const;

    inline double score(const ScanMatcherMap &map, const OrientedPoint &p, const double *readings) const;

    inline unsigned int likelihoodAndScore(double &s, double &l, const ScanMatcherMap &map, const OrientedPoint &p,
                                           const double *readings) const;

    double likelihood(double &lmax, OrientedPoint &mean, CovarianceMatrix &cov, const ScanMatcherMap &map,
                      const OrientedPoint &p, const double *readings);

    double likelihood(double &_lmax, OrientedPoint &_mean, CovarianceMatrix &_cov, const ScanMatcherMap &map,
                      const OrientedPoint &p, Gaussian3 &odometry, const double *readings, double gain = 180.);

    double closestMeanHitLikelihood(OrientedPoint &laser_pose, Point &end_point, double reading_range, double reading_bearing,
                                    const ScanMatcherMap &map, double &s, unsigned int &c) const;
    double measurementLikelihood(OrientedPoint &laser_pose, Point &end_point, double reading_range, double reading_bearing,
                                 const ScanMatcherMap &map, double &s, unsigned int &c) const;

    double overconfidenceUniformNoise(double l, bool out_of_range) const;

    inline const double *laserAngles() const { return m_laserAngles; }

    inline unsigned int laserBeams() const { return m_laserBeams; }

    static const double nullLikelihood;

  protected:
    // state of the matcher
    bool m_activeAreaComputed;

    /**laser parameters*/
    unsigned int m_laserBeams;
    double m_laserAngles[LASER_MAXBEAMS];
    // OrientedPoint m_laserPose;

    double computeCellR(const ScanMatcherMap &map, Point beamStart, Point beamEnd, IntPoint cell) const;

    PARAM_SET_GET(OrientedPoint, laserPose, protected, public, public)

    PARAM_SET_GET(double, laserMaxRange, protected, public, public)
    /**scan_matcher parameters*/
    PARAM_SET_GET(double, usableRange, protected, public, public)

    PARAM_SET_GET(double, gaussianSigma, protected, public, public)

    PARAM_SET_GET(double, likelihoodSigma, protected, public, public)

    PARAM_SET_GET(int, kernelSize, protected, public, public)

    PARAM_SET_GET(double, optAngularDelta, protected, public, public)

    PARAM_SET_GET(double, optLinearDelta, protected, public, public)

    PARAM_SET_GET(unsigned int, optRecursiveIterations, protected, public, public)

    PARAM_SET_GET(unsigned int, likelihoodSkip, protected, public, public)

    PARAM_SET_GET(double, llsamplerange, protected, public, public)

    PARAM_SET_GET(double, llsamplestep, protected, public, public)

    PARAM_SET_GET(double, lasamplerange, protected, public, public)

    PARAM_SET_GET(double, lasamplestep, protected, public, public)

    PARAM_SET_GET(bool, generateMap, protected, public, public)

    PARAM_SET_GET(ScanMatcherMap::MapModel, mapModel, protected, public, public)

    PARAM_SET_GET(ParticleWeighting, particleWeighting, protected, public, public)

    PARAM_SET_GET(double, enlargeStep, protected, public, public)

    PARAM_SET_GET(double, fullnessThreshold, protected, public, public)

    PARAM_SET_GET(double, angularOdometryReliability, protected, public, public)

    PARAM_SET_GET(double, linearOdometryReliability, protected, public, public)

    PARAM_SET_GET(double, freeCellRatio, protected, public, public)

    PARAM_SET_GET(unsigned int, initialBeamsSkip, protected, public, public)

    PARAM_SET_GET(double, overconfidenceUniformWeight, protected, public, public)

    // allocate this large array only once
    IntPoint *m_linePoints;
  };

  inline double ScanMatcher::icpStep(OrientedPoint &pret, const ScanMatcherMap &map, const OrientedPoint &p,
                                     const double *readings) const {
    const double *angle = m_laserAngles + m_initialBeamsSkip;
    OrientedPoint lp = p;
    lp.x += cos(p.theta) * m_laserPose.x - sin(p.theta) * m_laserPose.y;
    lp.y += sin(p.theta) * m_laserPose.x + cos(p.theta) * m_laserPose.y;
    lp.theta += m_laserPose.theta;
    unsigned int skip = 0;
    double freeDelta = map.getDelta() * m_freeCellRatio;
    std::list<PointPair> pairs;

    for (const double *r = readings + m_initialBeamsSkip; r < readings + m_laserBeams; r++, angle++) {
      skip++;
      skip = skip > m_likelihoodSkip ? 0 : skip;
      if (*r > m_usableRange || *r == 0.0)
        continue;
      if (skip)
        continue;
      Point phit = lp;
      phit.x += *r * cos(lp.theta + *angle);
      phit.y += *r * sin(lp.theta + *angle);
      IntPoint iphit = map.world2map(phit);
      Point pfree = lp;
      pfree.x += (*r - map.getDelta() * freeDelta) * cos(lp.theta + *angle);
      pfree.y += (*r - map.getDelta() * freeDelta) * sin(lp.theta + *angle);
      pfree = pfree - phit;
      IntPoint ipfree = map.world2map(pfree);
      bool found = false;
      Point bestMu(0., 0.);
      Point bestCell(0., 0.);
      for (int xx = -m_kernelSize; xx <= m_kernelSize; xx++)
        for (int yy = -m_kernelSize; yy <= m_kernelSize; yy++) {
          IntPoint pr = iphit + IntPoint(xx, yy);
          IntPoint pf = pr + ipfree;
          // AccessibilityState s=map.storage().cellState(pr);
          // if (s&Inside && s&Allocated){
          const PointAccumulator &cell = map.cell(pr);
          const PointAccumulator &fcell = map.cell(pf);
          if (((double)cell) > m_fullnessThreshold && ((double)fcell) < m_fullnessThreshold) {
            Point mu = phit - cell.mean();
            if (!found) {
              bestMu = mu;
              bestCell = cell.mean();
              found = true;
            } else if ((mu * mu) < (bestMu * bestMu)) {
              bestMu = mu;
              bestCell = cell.mean();
            }
          }
          //}
        }
      if (found) {
        pairs.push_back(std::make_pair(phit, bestCell));
        // std::cerr << "(" << phit.x-bestCell.x << "," << phit.y-bestCell.y << ") ";
      }
      // std::cerr << std::endl;
    }

    OrientedPoint result(0, 0, 0);
    // double icpError=icpNonlinearStep(result,pairs);
    std::cerr << "result(" << pairs.size() << ")=" << result.x << " " << result.y << " " << result.theta << std::endl;
    pret.x = p.x + result.x;
    pret.y = p.y + result.y;
    pret.theta = p.theta + result.theta;
    pret.theta = atan2(sin(pret.theta), cos(pret.theta));
    return score(map, p, readings);
  }

  inline double ScanMatcher::score(const ScanMatcherMap &map, const OrientedPoint &p, const double *readings) const {
    double s = 0;
    const double *angle = m_laserAngles + m_initialBeamsSkip;
    OrientedPoint lp = p;
    lp.x += cos(p.theta) * m_laserPose.x - sin(p.theta) * m_laserPose.y;
    lp.y += sin(p.theta) * m_laserPose.x + cos(p.theta) * m_laserPose.y;
    lp.theta += m_laserPose.theta;
    unsigned int skip = 0;
    double freeDelta = map.getDelta() * m_freeCellRatio;
    for (const double *r = readings + m_initialBeamsSkip; r < readings + m_laserBeams; r++, angle++) {
      skip++;
      skip = skip > m_likelihoodSkip ? 0 : skip;
      if (skip || *r > m_usableRange || *r == 0.0)
        continue;
      Point phit = lp;
      phit.x += *r * cos(lp.theta + *angle);
      phit.y += *r * sin(lp.theta + *angle);
      IntPoint iphit = map.world2map(phit);
      Point pfree = lp;
      pfree.x += (*r - map.getDelta() * freeDelta) * cos(lp.theta + *angle);
      pfree.y += (*r - map.getDelta() * freeDelta) * sin(lp.theta + *angle);
      pfree = pfree - phit;
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
            Point mu = phit - cell.mean();
            if (!found) {
              bestMu = mu;
              found = true;
            } else
              bestMu = (mu * mu) < (bestMu * bestMu) ? mu : bestMu;
          }
          //}
        }
      if (found)
        s += exp(-1. / m_gaussianSigma * bestMu * bestMu);
    }
    return s;
  }

  inline unsigned int ScanMatcher::likelihoodAndScore(double &s, double &l, const ScanMatcherMap &map, const OrientedPoint &p,
                                                      const double *readings) const {
    using namespace std;
    l = 0;
    s = 0;
    const double *angle = m_laserAngles + m_initialBeamsSkip;
    OrientedPoint lp = p;
    lp.x += cos(p.theta) * m_laserPose.x - sin(p.theta) * m_laserPose.y;
    lp.y += sin(p.theta) * m_laserPose.x + cos(p.theta) * m_laserPose.y;
    lp.theta += m_laserPose.theta;

    unsigned int skip = 0;
    unsigned int c = 0;

    for (const double *r = readings + m_initialBeamsSkip; r < readings + m_laserBeams; r++, angle++) {
      skip++;
      skip = skip > m_likelihoodSkip ? 0 : skip;
      if (skip)
        continue;

      double range = *r;
      double tmp_l;

      Point phit = lp;
      phit.x += range * cos(lp.theta + *angle);
      phit.y += range * sin(lp.theta + *angle);

      switch (m_particleWeighting) {
        case ClosestMeanHitLikelihood:
          tmp_l = closestMeanHitLikelihood(lp, phit, *r, *angle, map, s, c);
          break;
        case MeasurementLikelihood:
          tmp_l = measurementLikelihood(lp, phit, *r, *angle, map, s, c);
          break;
        case ForwardSensorModel:
          tmp_l = measurementLikelihood(lp, phit, *r, *angle, map, s, c);
          break;
      }

      if (m_particleWeighting == MeasurementLikelihood) {
        if (isnan(tmp_l)) {
          l = -std::numeric_limits<double>::max();
          break;
        }
      }

      l += tmp_l;
    }
    if (m_particleWeighting == MeasurementLikelihood)
      s = score(map, p, readings);

    return c;
  }

}  // namespace GMapping

#endif

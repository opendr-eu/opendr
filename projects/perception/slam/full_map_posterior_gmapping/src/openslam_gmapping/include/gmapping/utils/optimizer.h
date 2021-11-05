
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

#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include "gmapping/utils/point.h"

namespace GMapping {

  struct OptimizerParams {
    double discretization;
    double angularStep, linearStep;
    int iterations;
    double maxRange;
  };

  template<typename Likelihood, typename Map> struct Optimizer {
    Optimizer(const OptimizerParams &params);

    OptimizerParams params;
    Map lmap;
    Likelihood likelihood;

    OrientedPoint gradientDescent(const RangeReading &oldReading, const RangeReading &newReading);

    OrientedPoint gradientDescent(const RangeReading &oldReading, const OrientedPoint &pose, OLocalMap &Map);

    enum Move { Forward, Backward, Left, Right, TurnRight, TurnLeft };
  };

  template<typename Likelihood, typename Map>
  Optimizer<Likelihood, Map>::Optimizer(const OptimizerParams &p) : params(p), lmap(p.discretization) {}

  template<typename Likelihood, typename Map>
  OrientedPoint Optimizer<Likelihood, Map>::gradientDescent(const RangeReading &oldReading, const RangeReading &newReading) {
    lmap.clear();
    lmap.update(oldReading, OrientedPoint(0, 0, 0), params.maxRange);
    OrientedPoint delta = absoluteDifference(newReading.getPose(), oldReading.getPose());
    OrientedPoint bestPose = delta;
    double bestScore = likelihood(lmap, newReading, bestPose, params.maxRange);
    int it = 0;
    double lstep = params.linearStep, astep = params.angularStep;
    bool increase;
    /*	cerr << "bestScore=" << bestScore << endl;;*/
    do {
      increase = false;
      OrientedPoint itBestPose = bestPose;
      double itBestScore = bestScore;
      bool itIncrease;
      do {
        itIncrease = false;
        OrientedPoint testBestPose = itBestPose;
        double testBestScore = itBestScore;
        for (Move move = Forward; move <= TurnLeft; move = (Move)((int)move + 1)) {
          OrientedPoint testPose = itBestPose;
          switch (move) {
            case Forward:
              testPose.x += lstep;
              break;
            case Backward:
              testPose.x -= lstep;
              break;
            case Left:
              testPose.y += lstep;
              break;
            case Right:
              testPose.y -= lstep;
              break;
            case TurnRight:
              testPose.theta -= astep;
              break;
            case TurnLeft:
              testPose.theta += astep;
              break;
          }
          double score = likelihood(lmap, newReading, testPose, params.maxRange);
          if (score > testBestScore) {
            testBestScore = score;
            testBestPose = testPose;
          }
        }
        if (testBestScore > itBestScore) {
          itBestScore = testBestScore;
          itBestPose = testBestPose;
          /*				cerr << "s=" << itBestScore << " ";*/
          itIncrease = true;
        }
      } while (itIncrease);
      if (itBestScore > bestScore) {
        /*			cerr << "S(" << itBestScore << "," <<  bestScore<< ")";*/
        bestScore = itBestScore;
        bestPose = itBestPose;
        increase = true;
      } else {
        it++;
        lstep *= 0.5;
        astep *= 0.5;
      }
    } while (it < params.iterations);
    /*	cerr << "FinalBestScore" << bestScore << endl;*/
    cerr << endl;
    return bestPose;
  }

  template<typename Likelihood, typename Map>
  OrientedPoint Optimizer<Likelihood, Map>::gradientDescent(const RangeReading &reading, const OrientedPoint &pose,
                                                            OLocalMap &lmap) {
    OrientedPoint bestPose = pose;
    double bestScore = likelihood(lmap, reading, bestPose, params.maxRange);
    int it = 0;
    double lstep = params.linearStep, astep = params.angularStep;
    bool increase;
    /*	cerr << "bestScore=" << bestScore << endl;;*/
    do {
      increase = false;
      OrientedPoint itBestPose = bestPose;
      double itBestScore = bestScore;
      bool itIncrease;
      do {
        itIncrease = false;
        OrientedPoint testBestPose = itBestPose;
        double testBestScore = itBestScore;
        for (Move move = Forward; move <= TurnLeft; move = (Move)((int)move + 1)) {
          OrientedPoint testPose = itBestPose;
          switch (move) {
            case Forward:
              testPose.x += lstep;
              break;
            case Backward:
              testPose.x -= lstep;
              break;
            case Left:
              testPose.y += lstep;
              break;
            case Right:
              testPose.y -= lstep;
              break;
            case TurnRight:
              testPose.theta -= astep;
              break;
            case TurnLeft:
              testPose.theta += astep;
              break;
          }
          double score = likelihood(lmap, reading, testPose, params.maxRange);
          if (score > testBestScore) {
            testBestScore = score;
            testBestPose = testPose;
          }
        }
        if (testBestScore > itBestScore) {
          itBestScore = testBestScore;
          itBestPose = testBestPose;
          /*				cerr << "s=" << itBestScore << " ";*/
          itIncrease = true;
        }
      } while (itIncrease);
      if (itBestScore > bestScore) {
        /*			cerr << "S(" << itBestScore << "," <<  bestScore<< ")";*/
        bestScore = itBestScore;
        bestPose = itBestPose;
        increase = true;
      } else {
        it++;
        lstep *= 0.5;
        astep *= 0.5;
      }
    } while (it < params.iterations);
    /*	cerr << "FinalBestScore" << bestScore << endl;*/
    cerr << endl;
    return bestPose;
  }

}  // end namespace GMapping
#endif

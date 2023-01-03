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

#ifndef GFSREADER_H
#define GFSREADER_H

#include <gmapping/utils/point.h>
#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <vector>

#define MAX_LINE_LENGHT (1000000)

namespace GMapping {

  namespace GFSReader {

    using namespace std;

    struct Record {
      unsigned int dim;
      double time;

      virtual ~Record();

      virtual void read(istream &is) = 0;

      virtual void write(ostream &os);
    };

    struct CommentRecord : public Record {
      string text;

      virtual void read(istream &is);

      virtual void write(ostream &os);
    };

    struct PoseRecord : public Record {
      PoseRecord(bool ideal = false);

      void read(istream &is);

      virtual void write(ostream &os);

      bool truePos;
      OrientedPoint pose;
    };

    struct NeffRecord : public Record {
      void read(istream &is);

      virtual void write(ostream &os);

      double neff;
    };

    struct EntropyRecord : public Record {
      void read(istream &is);

      virtual void write(ostream &os);

      double poseEntropy;
      double trajectoryEntropy;
      double mapEntropy;
    };

    struct OdometryRecord : public Record {
      virtual void read(istream &is);

      vector<OrientedPoint> poses;
    };

    struct RawOdometryRecord : public Record {
      virtual void read(istream &is);

      OrientedPoint pose;
    };

    struct ScanMatchRecord : public Record {
      virtual void read(istream &is);

      vector<OrientedPoint> poses;
      vector<double> weights;
    };

    struct LaserRecord : public Record {
      virtual void read(istream &is);

      virtual void write(ostream &os);

      vector<double> readings;
      OrientedPoint pose;
      double weight;
    };

    struct ResampleRecord : public Record {
      virtual void read(istream &is);

      vector<unsigned int> indexes;
    };

    struct RecordList : public list<Record *> {
      mutable int sampleSize;

      istream &read(istream &is);

      double getLogWeight(unsigned int i) const;

      double getLogWeight(unsigned int i, RecordList::const_iterator frame) const;

      unsigned int getBestIdx() const;

      void printLastParticles(ostream &os) const;

      void printPath(ostream &os, unsigned int i, bool err = false, bool rawodom = false) const;

      RecordList computePath(unsigned int i, RecordList::const_iterator frame) const;

      void destroyReferences();
    };

  };  // end namespace GFSReader

};  // end namespace GMapping

#endif

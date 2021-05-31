# Copyright 2020-2021 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The code is based on KITTI tracking dev kit code (http://www.cvlibs.net/datasets/kitti/eval_tracking.php)

import copy
import math
import numpy as np
from typing import List
from munkres import Munkres
from collections import defaultdict
from opendr.engine.target import TrackingAnnotation3DList

from opendr.perception.object_detection_3d.voxel_object_detection_3d.second_detector.core.box_np_ops import (
    center_to_corner_box3d,
)
from opendr.perception.object_tracking_3d.ab3dmot.logger import Logger
from numba.cuda.cudadrv.error import CudaSupportError

try:
    from opendr.perception.object_detection_3d.voxel_object_detection_3d.\
        second_detector.core.non_max_suppression.nms_gpu import (
            rotate_iou_gpu_eval as iou3D
        )
except (CudaSupportError, ValueError):

    def iou3D(boxes, qboxes, criterion=-1):
        return np.ones((boxes.shape[0], qboxes.shape[0]))


class TrackingData:
    """
        Utility class to load data.
    """

    def __init__(
        self,
        frame=-1,
        obj_type="unset",
        truncation=-1,
        occlusion=-1,
        obs_angle=-10,
        x1=-1,
        y1=-1,
        x2=-1,
        y2=-1,
        w=-1,
        h=-1,
        length=-1,
        X=-1000,
        Y=-1000,
        Z=-1000,
        yaw=-10,
        score=-1000,
        track_id=-1,
    ):
        """
            Constructor, initializes the object given the parameters.
        """

        # init object data
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.truncation = truncation
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = w
        self.h = h
        self.length = length
        self.X = X
        self.Y = Y
        self.Z = Z
        self.yaw = yaw
        self.score = score
        self.ignored = False
        self.valid = False
        self.tracker = -1

    def __str__(self):
        """
            Print read data.
        """

        attrs = vars(self)
        return "\n".join("%s: %s" % item for item in attrs.items())


class TrackingEvaluator(object):
    """ tracking statistics (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
             MOTA	- Multi-object tracking accuracy in [0,100]
             MOTP	- Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
             MOTAL	- Multi-object tracking accuracy in [0,100] with log10(id-switches)

             id-switches - number of id switches
             fragments   - number of fragmentations

             MT, PT, ML	- number of mostly tracked, partially tracked and mostly lost trajectories

             recall	        - recall = percentage of detected targets
             precision	    - precision = percentage of correctly detected targets
             FAR		    - number of false alarms per frame
             falsepositives - number of false positives (FP)
             missed         - number of missed targets (FN)
    """

    def __init__(
        self,
        cls,
        n_sequences,
        n_frames,
        min_overlap=0.5,
        max_truncation=0,
        min_height=25,
        max_occlusion=2,
        log=print,
    ):

        self.n_sequences = n_sequences
        self.n_frames = n_frames
        self.log = log

        # class to evaluate, i.e. pedestrian or car
        self.cls = cls

        # statistics and numbers for evaluation
        self.n_gt = 0  # number of ground truth detections minus ignored false negatives and true positives
        self.n_igt = 0  # number of ignored ground truth detections
        self.n_gts = (
            []
        )  # number of ground truth detections minus ignored false negatives and true positives PER SEQUENCE
        self.n_igts = (
            []
        )  # number of ground ignored truth detections PER SEQUENCE
        self.n_gt_trajectories = 0
        self.n_gt_seq = []
        self.n_tr = (
            0  # number of tracker detections minus ignored tracker detections
        )
        self.n_trs = (
            []
        )  # number of tracker detections minus ignored tracker detections PER SEQUENCE
        self.n_itr = 0  # number of ignored tracker detections
        self.n_itrs = []  # number of ignored tracker detections PER SEQUENCE
        self.n_igttr = 0  # number of ignored ground truth detections
        # where the corresponding associated tracker detection is also ignored
        self.n_tr_trajectories = 0
        self.n_tr_seq = []
        self.MOTA = 0
        self.MOTP = 0
        self.MOTAL = 0
        self.MODA = 0
        self.MODP = 0
        self.MODP_t = []
        self.recall = 0
        self.precision = 0
        self.F1 = 0
        self.FAR = 0
        self.total_cost = 0
        self.itp = 0  # number of ignored true positives
        self.itps = []  # number of ignored true positives PER SEQUENCE
        self.tp = (
            0  # number of true positives including ignored true positives!
        )
        self.tps = (
            []
        )  # number of true positives including ignored true positives PER SEQUENCE
        self.fn = (
            0  # number of false negatives WITHOUT ignored false negatives
        )
        self.fns = (
            []
        )  # number of false negatives WITHOUT ignored false negatives PER SEQUENCE
        self.ifn = 0  # number of ignored false negatives
        self.ifns = []  # number of ignored false negatives PER SEQUENCE
        self.fp = 0  # number of false positives
        # a bit tricky, the number of ignored false negatives and ignored true positives
        # is subtracted, but if both tracker detection and ground truth detection
        # are ignored this number is added again to avoid double counting
        self.fps = []  # above PER SEQUENCE
        self.mme = 0
        self.fragments = 0
        self.id_switches = 0
        self.MT = 0
        self.PT = 0
        self.ML = 0

        self.min_overlap = (
            min_overlap  # minimum bounding box overlap for 3rd party metrics
        )
        self.max_truncation = (
            max_truncation  # maximum truncation of an object for evaluation
        )
        self.max_occlusion = (
            max_occlusion  # maximum occlusion of an object for evaluation
        )
        self.min_height = (
            min_height  # minimum height of an object for evaluation
        )
        self.n_sample_points = 500

        # this should be enough to hold all groundtruth trajectories
        # is expanded if necessary and reduced in any case
        self.gt_trajectories = [[] for x in range(self.n_sequences)]
        self.ign_trajectories = [[] for x in range(self.n_sequences)]

    def load_ground_truths(self, ground_truths: List[TrackingAnnotation3DList]):
        """
            Helper function to load ground truth.
        """

        try:
            self._load_data(
                ground_truths, cls=self.cls, loading_groundtruth=True
            )
        except IOError:
            return False
        return True

    def load_predictions(self, predictions: List[TrackingAnnotation3DList]):
        """
            Helper function to load tracker data.
        """

        try:
            if not self._load_data(
                predictions, cls=self.cls, loading_groundtruth=False
            ):
                return False
        except IOError:
            return False
        return True

    def _load_data(
        self, input_sequences: List[TrackingAnnotation3DList], cls, min_score=-1000, loading_groundtruth=False
    ):
        """
            Generic loader for ground truth and tracking data.
            Use loadGroundtruth() or loadTracker() to load this data.
            Loads detections in KITTI format from textfiles.
        """

        # classes that should be loaded (ignored neighboring classes)
        if "car" in cls.lower():
            classes = ["car", "van"]
        elif "pedestrian" in cls.lower():
            classes = ["pedestrian", "person_sitting"]
        else:
            classes = [cls.lower()]

        if loading_groundtruth:
            classes += ["dontcare"]

        # construct objectDetections object to hold detection data
        t_data = TrackingData()
        eval_2d = True
        eval_3d = True

        seq_data = []
        n_trajectories = 0
        n_trajectories_seq = []
        for seq, input_seq_data in enumerate(input_sequences):
            ids = []
            n_in_seq = 0
            id_frame_cache = []

            input_seq_boxes = []

            for boxList in input_seq_data:
                input_seq_boxes += boxList.boxes

            f_data = [[] for x in range(input_seq_boxes[-1].frame + 1)]

            for TrackingAnnotation3D in input_seq_boxes:
                # KITTI tracking benchmark data format:
                # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)

                if not any([s for s in classes if s == TrackingAnnotation3D.name.lower()]):
                    continue
                # get fields from table
                t_data.frame = int(TrackingAnnotation3D.frame)
                t_data.track_id = int(TrackingAnnotation3D.id)
                t_data.obj_type = TrackingAnnotation3D.name.lower()  # object type [car, pedestrian, cyclist, ...]
                t_data.truncation = int(
                    TrackingAnnotation3D.truncated
                )  # truncation [-1,0,1,2]
                t_data.occlusion = int(
                    TrackingAnnotation3D.occluded
                )  # occlusion  [-1,0,1,2]
                t_data.obs_angle = float(TrackingAnnotation3D.alpha)  # observation angle [rad]
                t_data.x1 = float(TrackingAnnotation3D.bbox2d[0])  # left   [px]
                t_data.y1 = float(TrackingAnnotation3D.bbox2d[1])  # top    [px]
                t_data.x2 = float(TrackingAnnotation3D.bbox2d[2])  # right  [px]
                t_data.y2 = float(TrackingAnnotation3D.bbox2d[3])  # bottom [px]
                t_data.h = float(TrackingAnnotation3D.dimensions[0])  # height [m]
                t_data.w = float(TrackingAnnotation3D.dimensions[1])  # width  [m]
                t_data.length = float(TrackingAnnotation3D.dimensions[2])  # length [m]
                t_data.X = float(TrackingAnnotation3D.location[0])  # X [m]
                t_data.Y = float(TrackingAnnotation3D.location[1])  # Y [m]
                t_data.Z = float(TrackingAnnotation3D.location[2])  # Z [m]
                t_data.yaw = float(TrackingAnnotation3D.rotation_y)  # yaw angle [rad]
                t_data.score = float(TrackingAnnotation3D.confidence)

                # do not consider objects marked as invalid
                if t_data.track_id is -1 and t_data.obj_type != "dontcare":
                    continue

                idx = t_data.frame
                # check if length for frame data is sufficient
                if idx >= len(f_data):
                    raise ValueError("Frame " + str(idx) + "is out of range")

                id_frame = (t_data.frame, t_data.track_id)
                if id_frame in id_frame_cache and not loading_groundtruth:
                    raise ValueError(
                        "track ids are not unique for sequence %d: frame %d"
                        % (seq, t_data.frame) + (
                            " track id %d occured at least twice for this frame"
                            % t_data.track_id
                        )
                    )
                id_frame_cache.append(id_frame)
                f_data[t_data.frame].append(copy.copy(t_data))

                if (
                    t_data.track_id not in ids and
                    t_data.obj_type != "dontcare"
                ):
                    ids.append(t_data.track_id)
                    n_trajectories += 1
                    n_in_seq += 1

                # check if uploaded data provides information for 2D and 3D evaluation
                if (
                    not loading_groundtruth and
                    eval_2d is True and
                    (
                        t_data.x1 == -1 or
                        t_data.x2 == -1 or
                        t_data.y1 == -1 or
                        t_data.y2 == -1
                    )
                ):
                    eval_2d = False
                if (
                    not loading_groundtruth and
                    eval_3d is True and
                    (
                        t_data.X == -1000 or
                        t_data.Y == -1000 or
                        t_data.Z == -1000
                    )
                ):
                    eval_3d = False

            # only add existing frames
            n_trajectories_seq.append(n_in_seq)
            seq_data.append(f_data)

        if not loading_groundtruth:
            self.tracker = seq_data
            self.n_tr_trajectories = n_trajectories
            self.eval_2d = eval_2d
            self.eval_3d = eval_3d
            self.n_tr_seq = n_trajectories_seq
            if self.n_tr_trajectories == 0:
                return False
        else:
            # split ground truth and DontCare areas
            self.dcareas = []
            self.groundtruth = []
            for seq_idx in range(len(seq_data)):
                seq_gt = seq_data[seq_idx]
                s_g, s_dc = [], []
                for f in range(len(seq_gt)):
                    all_gt = seq_gt[f]
                    g, dc = [], []
                    for gg in all_gt:
                        if gg.obj_type == "dontcare":
                            dc.append(gg)
                        else:
                            g.append(gg)
                    s_g.append(g)
                    s_dc.append(dc)
                self.dcareas.append(s_dc)
                self.groundtruth.append(s_g)
            self.n_gt_seq = n_trajectories_seq
            self.n_gt_trajectories = n_trajectories
        return True

    def box_overlap_3d(self, a: TrackingData, b: TrackingData, criterion="union"):
        """
            boxoverlap computes 3d intersection over union for bbox a and b in KITTI format.
            If the criterion is 'union', overlap = (a inter b) / a union b).
            If the criterion is 'a', overlap = (a inter b) / a, where b should be a dontcare area.
        """

        result = iou3D(
            center_to_corner_box3d(
                np.array([a.X, a.Y, a.Z]).reshape(1, -1),
                np.array([a.h, a.w, a.length]).reshape(1, -1),
                np.array([a.yaw])
            ),
            center_to_corner_box3d(
                np.array([b.X, b.Y, b.Z]).reshape(1, -1),
                np.array([b.h, b.w, b.length]).reshape(1, -1),
                np.array([b.yaw])
            )
        )

        return np.sum(result)

    def compute(self):
        """
            Computes the metrics defined in
                - Stiefelhagen 2008: Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics
                  MOTA, MOTAL, MOTP
                - Nevatia 2008: Global Data Association for Multi-Object Tracking Using Network Flows
                  MT/PT/ML
        """

        # construct Munkres object for Hungarian Method association
        hm = Munkres()
        max_cost = 1e9

        # go through all frames and associate ground truth and tracker results
        # groundtruth and tracker contain lists for every single frame containing lists of KITTI format detections
        fr, ids = 0, 0
        for seq_idx in range(len(self.groundtruth)):
            self.log(
                Logger.LOG_WHEN_NORMAL, "Computing metrics " + self.cls +
                "[" + str(seq_idx + 1) + "/" + str(len(self.groundtruth)) + "]", end='\r'
            )
            seq_gt = self.groundtruth[seq_idx]
            seq_dc = self.dcareas[seq_idx]  # don't care areas
            seq_tracker = self.tracker[seq_idx]
            seq_trajectories = defaultdict(list)
            seq_ignored = defaultdict(list)

            # statistics over the current sequence, check the corresponding
            # variable comments in __init__ to get their meaning
            seqtp = 0
            seqitp = 0
            seqfn = 0
            seqifn = 0
            seqfp = 0
            seqigt = 0
            seqitr = 0

            last_ids = [[], []]

            n_gts = 0
            n_trs = 0

            for f in range(len(seq_gt)):

                g = seq_gt[f]
                dc = seq_dc[f]

                t = seq_tracker[f]
                # counting total number of ground truth and tracker objects
                self.n_gt += len(g)
                self.n_tr += len(t)

                n_gts += len(g)
                n_trs += len(t)

                # use hungarian method to associate, using boxoverlap 0..1 as cost
                # build cost matrix
                cost_matrix = []
                this_ids = [[], []]
                for gg in g:
                    # save current ids
                    this_ids[0].append(gg.track_id)
                    this_ids[1].append(-1)
                    gg.tracker = -1
                    gg.id_switch = 0
                    gg.fragmentation = 0
                    cost_row = []
                    for tt in t:
                        # overlap == 1 is cost ==0
                        c = 1 - self.box_overlap_3d(gg, tt)
                        # gating for boxoverlap
                        if c <= self.min_overlap:
                            cost_row.append(c)
                        else:
                            cost_row.append(max_cost)  # = 1e9
                    cost_matrix.append(cost_row)
                    # all ground truth trajectories are initially not associated
                    # extend groundtruth trajectories lists (merge lists)
                    seq_trajectories[gg.track_id].append(-1)
                    seq_ignored[gg.track_id].append(False)

                if len(g) == 0:
                    cost_matrix = [[]]
                # associate
                association_matrix = hm.compute(cost_matrix)

                # tmp variables for sanity checks and MODP computation
                tmptp = 0
                tmpfp = 0
                tmpfn = 0
                tmpc = (
                    0  # this will sum up the overlaps for all true positives
                )
                tmpcs = [0] * len(
                    g
                )  # this will save the overlaps for all true positives
                # the reason is that some true positives might be ignored
                # later such that the corrsponding overlaps can
                # be subtracted from tmpc for MODP computation

                # mapping for tracker ids and ground truth ids
                for row, col in association_matrix:
                    # apply gating on boxoverlap
                    c = cost_matrix[row][col]
                    if c < max_cost:
                        g[row].tracker = t[col].track_id
                        this_ids[1][row] = t[col].track_id
                        t[col].valid = True
                        g[row].distance = c
                        self.total_cost += 1 - c
                        tmpc += 1 - c
                        tmpcs[row] = 1 - c
                        seq_trajectories[g[row].track_id][-1] = t[col].track_id

                        # true positives are only valid associations
                        self.tp += 1
                        tmptp += 1
                    else:
                        g[row].tracker = -1
                        self.fn += 1
                        tmpfn += 1

                # associate tracker and DontCare areas
                # ignore tracker in neighboring classes
                nignoredtracker = 0  # number of ignored tracker detections
                ignoredtrackers = dict()  # will associate the track_id with -1
                # if it is not ignored and 1 if it is
                # ignored;
                # this is used to avoid double counting ignored
                # cases, see the next loop

                for tt in t:
                    ignoredtrackers[tt.track_id] = -1
                    # ignore detection if it belongs to a neighboring class or is
                    # smaller or equal to the minimum height

                    tt_height = abs(tt.y1 - tt.y2)
                    if (
                        (self.cls == "car" and tt.obj_type == "van") or
                        (
                            self.cls == "pedestrian" and
                            tt.obj_type == "person_sitting"
                        ) or
                        tt_height <= self.min_height
                    ) and not tt.valid:
                        nignoredtracker += 1
                        tt.ignored = True
                        ignoredtrackers[tt.track_id] = 1
                        continue
                    for d in dc:
                        overlap = self.box_overlap_3d(tt, d, "a")
                        if overlap > 0.5 and not tt.valid:
                            tt.ignored = True
                            nignoredtracker += 1
                            ignoredtrackers[tt.track_id] = 1
                            break

                # check for ignored FN/TP (truncation or neighboring object class)
                ignoredfn = 0  # the number of ignored false negatives
                nignoredtp = 0  # the number of ignored true positives
                nignoredpairs = (
                    0  # the number of ignored pairs, i.e. a true positive
                )
                # which is ignored but where the associated tracker
                # detection has already been ignored

                gi = 0
                for gg in g:
                    if gg.tracker < 0:
                        if (
                            gg.occlusion > self.max_occlusion or
                            gg.truncation > self.max_truncation or
                            (self.cls == "car" and gg.obj_type == "van") or
                            (
                                self.cls == "pedestrian" and
                                gg.obj_type == "person_sitting"
                            )
                        ):
                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            ignoredfn += 1

                    elif gg.tracker >= 0:
                        if (
                            gg.occlusion > self.max_occlusion or
                            gg.truncation > self.max_truncation or
                            (self.cls == "car" and gg.obj_type == "van") or
                            (
                                self.cls == "pedestrian" and
                                gg.obj_type == "person_sitting"
                            )
                        ):

                            seq_ignored[gg.track_id][-1] = True
                            gg.ignored = True
                            nignoredtp += 1

                            # if the associated tracker detection is already ignored,
                            # we want to avoid double counting ignored detections
                            if ignoredtrackers[gg.tracker] > 0:
                                nignoredpairs += 1

                            # for computing MODP, the overlaps from ignored detections
                            # are subtracted
                            tmpc -= tmpcs[gi]
                    gi += 1

                # the below might be confusion, check the comments in __init__
                # to see what the individual statistics represent

                # correct TP by number of ignored TP due to truncation
                # ignored TP are shown as tracked in visualization
                tmptp -= nignoredtp

                # count the number of ignored true positives
                self.itp += nignoredtp

                # adjust the number of ground truth objects considered
                self.n_gt -= ignoredfn + nignoredtp

                # count the number of ignored ground truth objects
                self.n_igt += ignoredfn + nignoredtp

                # count the number of ignored tracker objects
                self.n_itr += nignoredtracker

                # count the number of ignored pairs, i.e. associated tracker and
                # ground truth objects that are both ignored
                self.n_igttr += nignoredpairs

                # false negatives = associated gt bboxes exceding association threshold + non-associated gt bboxes
                #
                tmpfn += len(g) - len(association_matrix) - ignoredfn
                self.fn += len(g) - len(association_matrix) - ignoredfn
                self.ifn += ignoredfn

                # false positives = tracker bboxes - associated tracker bboxes
                # mismatches (mme_t)
                tmpfp += (
                    len(t) -
                    tmptp -
                    nignoredtracker -
                    nignoredtp +
                    nignoredpairs
                )
                self.fp += (
                    len(t) -
                    tmptp -
                    nignoredtracker -
                    nignoredtp +
                    nignoredpairs
                )
                # tmpfp   = len(t) - tmptp - nignoredtp # == len(t) - (tp - ignoredtp) - ignoredtp
                # self.fp += len(t) - tmptp - nignoredtp

                # update sequence data
                seqtp += tmptp
                seqitp += nignoredtp
                seqfp += tmpfp
                seqfn += tmpfn
                seqifn += ignoredfn
                seqigt += ignoredfn + nignoredtp
                seqitr += nignoredtracker

                # sanity checks
                # - the number of true positives minues ignored true positives
                #   should be greater or equal to 0
                # - the number of false negatives should be greater or equal to 0
                # - the number of false positives needs to be greater or equal to 0
                #   otherwise ignored detections might be counted double
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false negatives (plus ignored ones)
                #   should match the total number of ground truth objects
                # - the number of counted true positives (plus ignored ones)
                #   and the number of counted false positives
                #   plus the number of ignored tracker detections should
                #   match the total number of tracker detections; note that
                #   nignoredpairs is subtracted here to avoid double counting
                #   of ignored detection sin nignoredtp and nignoredtracker
                if tmptp < 0:
                    raise NameError("Something went wrong! TP is negative")
                if tmpfn < 0:
                    raise NameError("Something went wrong! FN is negative")
                if tmpfp < 0:
                    raise NameError("Something went wrong! FP is negative")
                if tmptp + tmpfn is not len(g) - ignoredfn - nignoredtp:
                    raise NameError(
                        "Something went wrong! nGroundtruth is not TP+FN"
                    )
                if (
                    tmptp +
                    tmpfp +
                    nignoredtp +
                    nignoredtracker -
                    nignoredpairs
                    is not len(t)
                ):
                    raise NameError(
                        "Something went wrong! nTracker is not TP+FP"
                    )

                # check for id switches or fragmentations
                for i, tt in enumerate(this_ids[0]):
                    if tt in last_ids[0]:
                        idx = last_ids[0].index(tt)
                        tid = this_ids[1][i]
                        lid = last_ids[1][idx]
                        if tid != lid and lid != -1 and tid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].id_switch = 1
                                ids += 1
                        if tid != lid and lid != -1:
                            if g[i].truncation < self.max_truncation:
                                g[i].fragmentation = 1
                                fr += 1

                # save current index
                last_ids = this_ids
                # compute MOTP_t
                MODP_t = 1
                if tmptp != 0:
                    MODP_t = tmpc / float(tmptp)
                self.MODP_t.append(MODP_t)

            # remove empty lists for current gt trajectories
            self.gt_trajectories[seq_idx] = seq_trajectories
            self.ign_trajectories[seq_idx] = seq_ignored

            # gather statistics for "per sequence" statistics.
            self.n_gts.append(n_gts)
            self.n_trs.append(n_trs)
            self.tps.append(seqtp)
            self.itps.append(seqitp)
            self.fps.append(seqfp)
            self.fns.append(seqfn)
            self.ifns.append(seqifn)
            self.n_igts.append(seqigt)
            self.n_itrs.append(seqitr)

        # compute MT/PT/ML, fragments, idswitches for all groundtruth trajectories
        n_ignored_tr_total = 0
        for seq_idx, (seq_trajectories, seq_ignored) in enumerate(
            zip(self.gt_trajectories, self.ign_trajectories)
        ):
            if len(seq_trajectories) == 0:
                continue
            tmpMT, tmpML, tmpPT, tmpId_switches, tmpFragments = [0] * 5
            n_ignored_tr = 0
            for g, ign_g in zip(
                seq_trajectories.values(), seq_ignored.values()
            ):
                # all frames of this gt trajectory are ignored
                if all(ign_g):
                    n_ignored_tr += 1
                    n_ignored_tr_total += 1
                    continue
                # all frames of this gt trajectory are not assigned to any detections
                if all([this == -1 for this in g]):
                    tmpML += 1
                    self.ML += 1
                    continue
                # compute tracked frames in trajectory
                last_id = g[0]
                # first detection (necessary to be in gt_trajectories) is always tracked
                tracked = 1 if g[0] >= 0 else 0
                lgt = 0 if ign_g[0] else 1
                for f in range(1, len(g)):
                    if ign_g[f]:
                        last_id = -1
                        continue
                    lgt += 1
                    if (
                        last_id != g[f] and
                        last_id != -1 and
                        g[f] != -1 and
                        g[f - 1] != -1
                    ):
                        tmpId_switches += 1
                        self.id_switches += 1
                    if (
                        f < len(g) - 1 and
                        g[f - 1] != g[f] and
                        last_id != -1 and
                        g[f] != -1 and
                        g[f + 1] != -1
                    ):
                        tmpFragments += 1
                        self.fragments += 1
                    if g[f] != -1:
                        tracked += 1
                        last_id = g[f]
                # handle last frame; tracked state is handled in for loop (g[f]!=-1)
                if (
                    len(g) > 1 and
                    g[f - 1] != g[f] and
                    last_id != -1 and
                    g[f] != -1 and
                    not ign_g[f]
                ):
                    tmpFragments += 1
                    self.fragments += 1

                # compute MT/PT/ML
                tracking_ratio = tracked / float(len(g) - sum(ign_g))
                if tracking_ratio > 0.8:
                    tmpMT += 1
                    self.MT += 1
                elif tracking_ratio < 0.2:
                    tmpML += 1
                    self.ML += 1
                else:  # 0.2 <= tracking_ratio <= 0.8
                    tmpPT += 1
                    self.PT += 1

        if (self.n_gt_trajectories - n_ignored_tr_total) == 0:
            self.MT = 0.0
            self.PT = 0.0
            self.ML = 0.0
        else:
            self.MT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.PT /= float(self.n_gt_trajectories - n_ignored_tr_total)
            self.ML /= float(self.n_gt_trajectories - n_ignored_tr_total)

        # precision/recall etc.
        if (self.fp + self.tp) == 0 or (self.tp + self.fn) == 0:
            self.recall = 0.0
            self.precision = 0.0
        else:
            self.recall = self.tp / float(self.tp + self.fn)
            self.precision = self.tp / float(self.fp + self.tp)
        if (self.recall + self.precision) == 0:
            self.F1 = 0.0
        else:
            self.F1 = (
                2.0 *
                (self.precision * self.recall) /
                (self.precision + self.recall)
            )
        if sum(self.n_frames) == 0:
            self.FAR = "n/a"
        else:
            self.FAR = self.fp / float(sum(self.n_frames))

        # compute CLEARMOT
        if self.n_gt == 0:
            self.MOTA = -float("inf")
            self.MODA = -float("inf")
        else:
            self.MOTA = 1 - (self.fn + self.fp + self.id_switches) / float(
                self.n_gt
            )
            self.MODA = 1 - (self.fn + self.fp) / float(self.n_gt)
        if self.tp == 0:
            self.MOTP = float("inf")
        else:
            self.MOTP = self.total_cost / float(self.tp)
        if self.n_gt != 0:
            if self.id_switches == 0:
                self.MOTAL = 1 - (
                    self.fn + self.fp + self.id_switches
                ) / float(self.n_gt)
            else:
                self.MOTAL = 1 - (
                    self.fn + self.fp + math.log10(self.id_switches)
                ) / float(self.n_gt)
        else:
            self.MOTAL = -float("inf")
        if sum(self.n_frames) == 0:
            self.MODP = "n/a"
        else:
            self.MODP = sum(self.MODP_t) / float(sum(self.n_frames))
        return True

    def create_summary(self):
        """
            Generate and mail a summary of the results.
            If mailpy.py is present, the summary is instead printed.
        """

        summary = ""

        summary += "tracking evaluation summary".center(80, "=") + "\n"
        summary += (
            self.print_entry(
                "Multiple Object Tracking Accuracy (MOTA)", self.MOTA
            ) +
            "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Tracking Precision (MOTP)", self.MOTP
            ) +
            "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Tracking Accuracy (MOTAL)", self.MOTAL
            ) +
            "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Detection Accuracy (MODA)", self.MODA
            ) +
            "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Detection Precision (MODP)", self.MODP
            ) +
            "\n"
        )
        summary += "\n"
        summary += self.print_entry("Recall", self.recall) + "\n"
        summary += self.print_entry("Precision", self.precision) + "\n"
        summary += self.print_entry("F1", self.F1) + "\n"
        summary += self.print_entry("False Alarm Rate", self.FAR) + "\n"
        summary += "\n"
        summary += self.print_entry("Mostly Tracked", self.MT) + "\n"
        summary += self.print_entry("Partly Tracked", self.PT) + "\n"
        summary += self.print_entry("Mostly Lost", self.ML) + "\n"
        summary += "\n"
        summary += self.print_entry("True Positives", self.tp) + "\n"
        summary += self.print_entry("Ignored True Positives", self.itp) + "\n"
        summary += self.print_entry("False Positives", self.fp) + "\n"
        summary += self.print_entry("False Negatives", self.fn) + "\n"
        summary += self.print_entry("Ignored False Negatives", self.ifn) + "\n"
        summary += self.print_entry("Missed Targets", self.fn) + "\n"
        summary += self.print_entry("ID-switches", self.id_switches) + "\n"
        summary += self.print_entry("Fragmentations", self.fragments) + "\n"
        summary += "\n"
        summary += (
            self.print_entry(
                "Ground Truth Objects (Total)", self.n_gt + self.n_igt
            ) +
            "\n"
        )
        summary += (
            self.print_entry("Ignored Ground Truth Objects", self.n_igt) + "\n"
        )
        summary += (
            self.print_entry(
                "Ground Truth Trajectories", self.n_gt_trajectories
            ) +
            "\n"
        )
        summary += "\n"
        summary += self.print_entry("Tracker Objects (Total)", self.n_tr) + "\n"
        summary += (
            self.print_entry("Ignored Tracker Objects", self.n_itr) + "\n"
        )
        summary += (
            self.print_entry("Tracker Trajectories", self.n_tr_trajectories) +
            "\n"
        )
        summary += "=" * 80

        return summary

    def print_entry(self, key, val, width=(70, 10)):
        """
            Pretty print an entry in a table fashion.
        """

        s_out = key.ljust(width[0])
        if type(val) == int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val) == float:
            s = "%%%df" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s" % val).rjust(width[1])
        return s_out


def evaluate(
    predictions: List[List[TrackingAnnotation3DList]],
    ground_truths: List[List[TrackingAnnotation3DList]],
    log=print
):

    results = {
        "car": None,
        "pedestrian": None,
        "cyclist": None,
    }

    for c in ("car", "pedestrian", "cyclist"):
        evaluator = TrackingEvaluator(
            cls=c, n_sequences=len(ground_truths),
            n_frames=[len(a) for a in ground_truths],
            log=log,
        )
        # load tracker data and check provided classes

        evaluator.load_predictions(predictions)
        evaluator.load_ground_truths(ground_truths)

        if len(evaluator.groundtruth) is not len(evaluator.tracker):
            raise ValueError(
                "The predictions data does not provide results for every sequence."
            )
            return results
        evaluator.compute()

        results[c] = evaluator.create_summary()

    return results

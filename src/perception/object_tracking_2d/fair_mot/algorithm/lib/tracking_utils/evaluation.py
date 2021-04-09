import copy
import motmetrics as mm
import numpy as np


mm.lap.default_solver = "lap"


class Evaluator(object):
    def __init__(self):
        self.reset_accumulator()

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, gt_tlwhs, gt_ids, trk_tlwhs, trk_ids, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        iou_distance = mm.distances.iou_matrix(
            gt_tlwhs, trk_tlwhs, max_iou=0.5
        )

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if (
            rtn_events
            and iou_distance.size > 0
            and hasattr(self.acc, "last_mot_events")
        ):
            events = (
                self.acc.last_mot_events
            )  # only supported by https://github.com/longcw/py-motmetrics
        else:
            events = None
        return events

    def eval_all(self, gt_tlwhs, gt_ids, trk_tlwhs, trk_ids):
        self.reset_accumulator()

        for frame_id in range(len(gt_ids)):
            self.eval_frame(
                frame_id, gt_tlwhs[frame_id], gt_ids[frame_id],
                trk_tlwhs[frame_id], trk_ids[frame_id],
                rtn_events=False
            )

        return self.acc

    @staticmethod
    def get_summary(
        accs,
        names,
        metrics=(
            "mota",
            "num_switches",
            "idp",
            "idr",
            "idf1",
            "precision",
            "recall",
        ),
    ):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs, metrics=metrics, names=names, generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd

        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()

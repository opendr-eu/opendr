"""
Modified based on: https://github.com/open-mmlab/mmskeleton
"""
from collections import OrderedDict
import continual as co

from opendr.perception.skeleton_based_action_recognition.algorithm.models.co_base import (
    CoModelBase,
    CoSpatioTemporalBlock,
)


class CoStGcnMod(CoModelBase):
    def __init__(
        self,
        num_point=25,
        num_person=2,
        in_channels=3,
        graph_type="ntu",
        sequence_len: int = 300,
        num_classes: int = 60,
        loss_name="cross_entropy",
    ):
        CoModelBase.__init__(
            self, num_point, num_person, in_channels, graph_type, sequence_len, num_classes, loss_name
        )

        # Shapes: num_channels, num_frames, num_vertices, num_skeletons
        (C_in, T, _, _) = self.input_shape
        A = self.graph.A

        # Pass in precise window-sizes to compensate propperly in BatchNorm modules
        # fmt: off
        self.layers = co.Sequential(OrderedDict([
            ("layer1", CoSpatioTemporalBlock(C_in, 64, A, padding=0, window_size=T, residual=False)),
            ("layer2", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 1 * 8)),
            ("layer3", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 2 * 8)),
            ("layer4", CoSpatioTemporalBlock(64, 64, A, padding=0, window_size=T - 3 * 8)),
            ("layer5", CoSpatioTemporalBlock(64, 128, A, padding=0, window_size=T - 4 * 8, stride=1)),
            ("layer6", CoSpatioTemporalBlock(128, 128, A, padding=0, window_size=(T - 4 * 8) / 2 - 1 * 8)),
            ("layer7", CoSpatioTemporalBlock(128, 128, A, padding=0, window_size=(T - 4 * 8) / 2 - 2 * 8)),
            ("layer8", CoSpatioTemporalBlock(128, 256, A, padding=0, window_size=(T - 4 * 8) / 2 - 3 * 8, stride=1)),
            ("layer9", CoSpatioTemporalBlock(256, 256, A, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 1 * 8)),
            ("layer10", CoSpatioTemporalBlock(256, 256, A, padding=0, window_size=((T - 4 * 8) / 2 - 3 * 8) / 2 - 2 * 8)),
        ]))
        # fmt: on

        # Other layers defined in CoModelBase.on_init_end
        CoModelBase.on_init_end(self)

# Copyright 2020-2023 OpenDR European Project
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

import numpy as np
from opendr.utils.ambiguity_measure.persistence import get_persistence
from opendr.engine.data import Image
from matplotlib import pyplot as plt, transforms, cm
from copy import deepcopy
from typing import Optional, Union, List


class AmbiguityMeasure(object):
    """
    AmbiguityMeasure tool.

    This tool can be used to obtain an ambiguity measure of the output of vision-based manipulation models, such as
    Transporter Nets and CLIPort.
    """

    def __init__(self, threshold: float = 0.5, temperature: float = 1.0):
        """
        Constructor of AmbiguityMeasure

        :param threshold: Ambiguity threshold, should be in [0, 1).
        :type threshold: float
        :param temperature: Temperature of the sigmoid function.
        :type temperature: float
        """
        assert threshold >= 0 < 1, "Threshold should be in [0, 1)."
        assert temperature > 0, "Temperature should be greater than 0."
        self._threshold = threshold
        self._temperature = temperature

    def get_ambiguity_measure(self, heatmap: np.ndarray):
        """
        Get Ambiguity Measure.

        :param heatmap: Pixel-wise value estimates.
        :type heatmap: np.ndarray
        :return: Tuple[ambiguous, locs, maxima, probs]
            - ambiguous: Whether or not output was ambiguous.
            - locs: Pixel locations of significant local maxima.
            - maxima: Values corresponding to local maxima.
            - probs: Probability mass function based on local maxima.
        :rtype: Tuple[ambiguous, locs, maxima, probs]
            - ambiguous: bool
            - locs: list
            - maxima: list
            - probs: list
        """
        # Calculate persistence to find local maxima
        persistence = get_persistence(heatmap)

        maxima = []
        locs = []
        for i, homclass in enumerate(persistence):
            p_birth, _, _, _ = homclass
            locs.append(p_birth)
            maxima.append(heatmap[p_birth[0], p_birth[1]])
        probs = self.__softmax(np.asarray(maxima))
        ambiguous = 1.0 - max(probs) < self._threshold
        return ambiguous, locs, maxima, probs

    def plot_ambiguity_measure(
        self,
        heatmap: np.ndarray,
        locs: List[List[int]],
        probs: Union[List[float], np.ndarray],
        img: Image = None,
        img_offset: float = -250.0,
        view_init: List[int] = [30, 30],
        plot_threshold: float = 0.05,
        title: str = "Ambiguity Measure",
        save_path: Optional[str] = None,
    ):
        """
        Plot the obtained ambiguity measure.

        :param heatmap: Pixel-wise value estimates.
        :type heatmap: np.ndarray
        :param locs: Pixel locations of significant local maxima.
        :type locs: List[List[int]]
        :param probs: Probability mass function based on local maxima.
        :type probs: List[float]
        :param img: Top view input image.
        :type img: Union[np.ndarray, Image]
        :param img_offset: Specifies the distance between value estimates and image.
        :type img_offset: float
        :param view_init: Set the elevation and azimuth of the axes in degrees (not radians).
        :type view_init: List[float]
        :param plot_threshold: Threshold for plotting probabilities.
        Probabilities lower than this value will not be plotted.
        :param title: Title of the plot.
        :type title: str
        :param save_path: Path for saving figure, if None,
        :type plot_threshold: float
        """
        fig = plt.figure()
        ax = plt.axes(projection="3d")
        ax.computed_zorder = False
        trans_offset = transforms.offset_copy(ax.transData, fig=fig, y=2, units="dots")
        X, Y = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
        Z = heatmap
        ax.set_title(title)
        ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False, shade=False, zorder=-1)

        if img is not None:
            if type(img) is Image:
                img = np.moveaxis(img.numpy(), 0, -1)

            img = deepcopy(img)
            if np.max(img) > 1:
                img = img / 255
            x_image, y_image = np.mgrid[0:img.shape[0], 0:img.shape[1]]
            ax.plot_surface(
                x_image,
                y_image,
                np.ones(img.shape[:2]) * -img_offset,
                rstride=1,
                cstride=1,
                facecolors=img,
                shade=False,
            )

        ax.set_zlim(-img_offset - 1, 50)
        ax.view_init(view_init[0], view_init[1])
        for loc, value in zip(locs, probs):
            if value > plot_threshold:
                ax.plot3D([loc[0]], [loc[1]], [value], "r.", zorder=9)
                ax.plot3D([loc[0]], [loc[1]], [-img_offset], "r.", zorder=-2)
                ax.text(
                    loc[0],
                    loc[1],
                    value,
                    f"{value:.2f}",
                    zorder=10,
                    transform=trans_offset,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    c="r",
                    fontsize="large",
                )
        ax.grid(False)
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @property
    def threshold(self):
        """
        Getter of threshold.

        :return: Threshold value.
        :rtype: float
        """
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        """
        Setter of threshold.

        :param threshold: Threshold value.
        :type threshold: float
        """
        if type(value) != float:
            raise TypeError("threshold should be a float")
        else:
            self._threshold = value

    def __softmax(self, x):
        x /= self._temperature
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

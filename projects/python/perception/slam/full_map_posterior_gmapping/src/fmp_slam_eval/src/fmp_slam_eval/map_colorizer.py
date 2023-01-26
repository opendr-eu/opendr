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

import matplotlib as mpl
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm, BoundaryNorm

from fmp_slam_eval.enums import DiscreteStates as DiSt


class MapColorizer:
    """
    Class for converting a floating point valued map into an RGBA image or a matplotlib figure.
    The maps can be a list, np.ndarray or, preferably, numpy masked arrays in order to support both continuous floating
    point values, as well as discrete states such as Undefined, Bimodal, etc.
    Such Discrete values are defined in disc_states.py, along with their values, colors and labels.
    """

    def __init__(self, wm_extent=None, ds_list=None):
        """
        Constructor. Initializes the MapColorizer Object.

        :param wm_extent: (list)[Default: None] World map extension in world units.
                                                Used for the values of the image ticks.
                                                E.g.: [x0, x1, y0, y1]

        :param ds_list: (list)[Default: None] List of possible discrete state values.
                                              Used for drawing the discrete color bar. All states listed will be
                                              displayed in the bar, even if the image has no pixels with that value.
                                              E.g.: [DiscreteStates.UNIFORM, DiscreteStates.UNDEFINED]
        """

        self._wm_extent = None
        self._aspect_ratio = 'equal'

        self._img_origin = 'upper'

        if wm_extent is None:
            wm_extent = [0, 100, 0, 100]

        self.set_wm_extent(wm_extent)

        if ds_list is None:
            ds_list = DiSt.list_all()

        self._cb_orientation = 'horizontal'

        # Discrete State Parameters
        self._ds_list = None

        self._clr_ds = None
        self._cmp_ds = None
        self._tks_ds = None
        self._tlb_ds = None
        self._bnd_ds = None
        self._nrm_ds = None
        self._map_ds = None

        self.set_disc_state_list(ds_list)

        # Color range
        self._v_min = 0
        self._v_max = 1

        self._cb_tick_count = 10

        # Continuous Interval Parameters
        self._cmp_ci = None
        self._tks_ci = None
        self._tlb_ci = None
        self._nrm_ci = None
        self._map_ci = None

        self._cb_ci_extend = 'neither'

        self.set_cont_bounds(None, self._v_min, self._v_max, occupancy_map=True)

    def set_wm_extent(self, wm_extent):
        """
        Set the world map extension.

        :param wm_extent: (list) World map extension in world units. Used for the values of the image ticks.
                                 E.g.: [x0, x1, y0, y1]
        :return: None
        """

        self._wm_extent = wm_extent

    def set_cb_orientation(self, orientation):
        """
        Set the color bar orientation in the plots

        :param orientation: ('horizontal'|'vertical')(Default: 'horizontal')

        :return: None
        """

        if orientation in ['horizontal', 'vertical']:
            self._cb_orientation = orientation
        else:
            self._cb_orientation = 'horizontal'

    def set_aspect_ratio(self, aspect_ratio):
        """
        Set the aspect ratio of the plots. Probably useless given that wm_extent is also in use.

        :param aspect_ratio: (string|float) Aspect ratio between the x and y axes of the plot

        :return: None
        """

        self._aspect_ratio = aspect_ratio

    def set_cb_tick_count(self, cb_tick_count):
        """
        Set the number of ticks in the continuous color bar

        :param cb_tick_count: (int) Number of ticks to divide the (v_min, v_max) interval of the continuous colorbar.

        :return: None
        """

        if cb_tick_count > 0:
            self._cb_tick_count = cb_tick_count
        else:
            self._cb_tick_count = 10

    def set_disc_state_list(self, ds_list):
        """
        Set the list of possible discrete states.
        Used for defining the possible values and colors, color map and normalization for the actual plot and also the
        color map, ticks and tick labels of the discrete color bar.

        :param ds_list: (list) List of possible discrete state values.
                               Used for drawing the discrete color bar. All states listed will be displayed in the bar,
                               even if the image has no pixels with that value.
                               E.g.: [DiscreteStates.UNIFORM, DiscreteStates.UNDEFINED]

        :return: None
        """

        if not ds_list:
            self._ds_list = []
            self._clr_ds = []
            self._cmp_ds = []
            self._tks_ds = []
            self._tlb_ds = []
            self._bnd_ds = []
            self._nrm_ds = []
            self._map_ds = []
            return

        self._ds_list = DiSt.sort_ds_list(ds_list)

        self._clr_ds = DiSt.get_colors(ds_list)    # Color list for discrete states
        self._cmp_ds = mpl.colors.ListedColormap(self._clr_ds, name="cm_ds")  # Colormap for discrete states

        self._tks_ds = [float(i) for i in range(len(ds_list))]  # Tick Values for discrete states
        self._tlb_ds = DiSt.get_labels(ds_list)  # Tick Labels for discrete states

        # Nasty fix for single discrete value
        if len(ds_list) == 1:
            tk_0 = self._tks_ds[0]
            self._tks_ds = [tk_0 - 1 + i for i in range(3)]
            self._tlb_ds = ['', self._tlb_ds[0], '']
            self._bnd_ds = np.array([tk_0 - 0.5 + i for i in range(4)])

        elif len(ds_list) == 2:
            tk_0 = self._tks_ds[0]
            tk_1 = self._tks_ds[1]
            tk_avg = (tk_0 + tk_1)/2
            self._tks_ds = [tk_0, tk_avg, tk_1]
            self._tlb_ds = [self._tlb_ds[0], '', self._tlb_ds[1]]
            self._bnd_ds = [tk_0 - 0.5, tk_avg, tk_1 + 0.5]

        else:
            self._bnd_ds = np.array(self._tks_ds) + 0.5
            self._bnd_ds = np.append(min(self._tks_ds) - 0.5, self._bnd_ds)  # Boundaries for the colors in the

        self._nrm_ds = BoundaryNorm(self._tks_ds, len(self._tks_ds))

        # Scalar Mappable for discrete state color bar
        self._map_ds = mpl.cm.ScalarMappable(
            cmap=self._cmp_ds,
            norm=plt.Normalize(vmin=min(self._tks_ds), vmax=max(self._tks_ds))
        )
        self._map_ds._A = []

    def set_cont_bounds(self, img, v_min=0, v_max=1, occupancy_map=True, log_scale=False):
        """
        Set the min and max continuous values that a certain cell can take.
        Used for Used for defining the possible values, color map and normalization for the actual plot and also the
        color map, ticks and tick labels for the continuous color bar.

        :param img: (numpy.ma|numpy.ndarray) Actual image to be plotted. Used only to determin its min and max values
                                             in case v_min, v_max or both are defined as None.
        :param v_min: (float|None)[Default: 0] Minimum value the map can take. If None, it will be taken as img.min()
        :param v_max: (float|None)[Default: 1] Maximum value the map can take. If None, it will be taken as img.max()
        :param occupancy_map: (bool)[Default: True] If True, then 'Free' and 'Occ' will be appended to the first and
                                                    last tick labels respectively if they were not defined as None.
        :param log_scale: (bool)[Default: False] Assign the colors logarithmically

        :return: None
        """

        self._cb_ci_extend = 'neither'

        if v_min is not None:
            self._v_min = v_min
        else:
            self._v_min = img.min()
            self._cb_ci_extend = 'min'

        if log_scale and self._v_min == 0:
            min_val = img.min()

            if min_val > 0:
                self._v_min = img.min()
            else:
                self._v_min = 10 ** -3

        if v_max is not None:
            self._v_max = v_max
        else:
            self._v_max = img.max()
            self._cb_ci_extend = 'max'

        if v_min is None and v_max is None:
            self._cmp_ci = mpl.cm.get_cmap('RdYlBu')
            self._cb_ci_extend = 'both'

        else:
            self._cmp_ci = mpl.cm.get_cmap('plasma_r')

        if log_scale:
            self._tks_ci = np.logspace(np.log10(self._v_min), np.log10(self._v_max), self._cb_tick_count)
            self._tlb_ci = list(np.char.mod('%.2E', self._tks_ci))

            self._nrm_ci = LogNorm(vmin=self._v_min, vmax=self._v_max)
        else:
            self._tks_ci = np.linspace(self._v_min, self._v_max, self._cb_tick_count)
            self._tlb_ci = list(np.char.mod('%.2f', self._tks_ci))

            self._nrm_ci = Normalize(vmin=self._v_min, vmax=self._v_max)

        self._map_ci = mpl.cm.ScalarMappable(
            cmap=self._cmp_ci,
            norm=self._nrm_ci
        )
        self._map_ci._A = []

        if occupancy_map:
            if v_min == 0:
                self._tlb_ci[0] += '\nFree'
            if v_max == 1:
                self._tlb_ci[-1] += '\nOcc'

    def _draw_cb(self, fig, mappable, params, tick_labels=None):
        """
        Draw a color bar.

        :param fig: (matplotlib.fig) Matplotlib Figure to add the color bar to.
        :param mappable: (matplotlib.cm.ScalarMappable) A ScalarMappable (independent from any actual plot) to use
                                                        as the color scale.
        :param params: (dict) Dictionary of arguments supported by plt.colorbar()
        :param tick_labels: (list|None)[Default: None] list of string labels for each tick in the color bar

        :return: The color bar object.
        """

        cb = fig.colorbar(mappable, **params)

        if tick_labels is not None:
            cb.set_ticklabels(tick_labels)
            if self._cb_orientation == 'horizontal':
                cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation='vertical')

        return cb

    def draw_cb_disc(self, fig):
        """
        Draw a Discretely Valued color bar

        :param fig: (matplotlib.fig) Matplotlib Figure to add the color bar to.

        :return: The discrete color bar object.
        """

        if not self._ds_list:
            return None

        cb_params = {
            'cmap':        self._cmp_ds,
            'ticks':       self._tks_ds,
            'boundaries':  self._bnd_ds,
            'norm':        self._nrm_ds,
            'orientation': self._cb_orientation,
            'spacing':     'uniform',  # 'proportional'
        }

        return self._draw_cb(fig, self._map_ds, cb_params, self._tlb_ds)

    def draw_cb_cont(self, fig):
        """
        Draw a Continuously Valued color bar

        :param fig: (matplotlib.fig) Matplotlib Figure to add the color bar to.

        :return: The continuous color bar object.
        """

        cb_params = {
            'cmap': self._map_ci,
            'ticks': self._tks_ci,
            'label': "Occupancy",
            'orientation': self._cb_orientation,
            'norm': self._nrm_ci,
            'extend': self._cb_ci_extend
        }

        return self._draw_cb(fig, self._map_ci, cb_params, self._tlb_ci)

    def _imshow_disc_map(self, ax, img):
        """
        Draw the discrete portion of a map as a colored image.

        :param ax: (matplotlib.ax) The Matplotlib axes object to plot to.
        :param img: (numpy.ma) A masked array representing a map with discrete values.

        :return: The actual image plot object.
        """

        params = {
            'cmap': self._cmp_ds,
            'extent': self._wm_extent,
            'vmin': min(self._tks_ds),
            'vmax': max(self._tks_ds),
            'origin': self._img_origin
        }

        return ax.imshow(img, **params)

    def _imshow_cont_map(self, ax, img):
        """
        Draw the continuous portion of a map as a colored image.

        :param ax: (matplotlib.ax) The Matplotlib axes object to plot to.
        :param img: (numpy.ma) A masked array representing a map with continuous values.

        :return: The actual image plot object.
        """

        params = {
            'cmap': self._cmp_ci,
            'extent': self._wm_extent,
            'vmin': self._v_min,
            'vmax': self._v_max,
            'origin': self._img_origin
        }

        return ax.imshow(img, **params)

    def _make_figure(self):
        """
        Creates a Matplotlib figure.

        :return: A tuple of empty Matplotlib figure and axes objects.
        """

        fig, ax = plt.subplots(figsize=[20, 20])

        ax.set_aspect(self._aspect_ratio)

        return fig, ax

    def _draw_plot(self, cont_map, ds_map=None, ds_list=None, v_min=0, v_max=1, occupancy_map=True):
        """
        Create a figure, draw the discrete map and color bar if not None, and then the continuous map and color bar.
        The figure isn't actually displayed, in case it is to be directly saved to a file.

        :param cont_map: (numpy.ma) A masked array representing a map with continuous values.
        :param ds_map: (numpy.ma)[Default: None] A masked array representing a map with discrete values.
                                                 If None, then only the continuous part will be drawn.
        :param v_min: (float|None)[Default: 0] Minimum value the map can take.
                                               If None, it will be taken as cont_map.min().
        :param v_max: (float|None)[Default: 1] Maximum value the map can take.
                                               If None, it will be taken as cont_map.max().
        :param occupancy_map: (bool)[Default: True] If True, then 'Free' and 'Occ' will be appended to the first and
                                                    last tick labels respectively if they were not defined as None.

        :return: A tuple of drawn Matplotlib figure and axes objects
        """

        fig, ax = self._make_figure()

        if ds_map is not None:
            if ds_list is not None:
                self.set_disc_state_list(ds_list)
            self._imshow_disc_map(ax, ds_map)
            self.draw_cb_disc(fig)

        self.set_cont_bounds(cont_map, v_min=v_min, v_max=v_max, occupancy_map=occupancy_map)
        self._imshow_cont_map(ax, cont_map)
        self.draw_cb_cont(fig)

        return fig, ax

    def colorize(self, cont_map, ds_map=None):
        """
        Generate an RGBa [0, 1] image from a continuous and discrete map without actually plotting it using Matplotlib.

        :param cont_map: (numpy.ma) A masked array representing a map with continuous values.
        :param ds_map: (numpy.ma)[Default: None] A masked array representing a map with discrete values.
                                                 If None, then only the continuous part will be drawn.

        :return: An RGBa (w, h, 4) -> [0, 1] image.
        """

        shape = cont_map.shape
        shape = (shape[0], shape[1], 4)
        rgba_img = np.zeros(shape)

        if ds_map is not None:
            # Remap Discrete States from enum values to equally spaced values for nicer colorbars.
            # E.g.: for [UNDEFINED, UNIFORM, ZERO] with values [1, 2, 4], remap to [0, 1, 2]
            # for evenly distributed ticks
            for i, ds_state_val in enumerate(self._ds_list):
                ds_map[ds_map == ds_state_val.value] = i
            # Then normalize to [0, 1] interval
            ds_map = self._nrm_ds(ds_map)
            # Anc colorize to RGB
            rgba_img += self._cmp_ds(ds_map)

        cont_map = self._nrm_ci(cont_map)
        rgba_img += self._cmp_ci(cont_map)

        return rgba_img

    def plot(self, cont_map, ds_map=None, v_min=0, v_max=1, occupancy_map=True):
        """
        Create and display a figure, draw the discrete map and color bar if not None, and then the continuous map and
        color bar.

        :param cont_map: (numpy.ma) A masked array representing a map with continuous values.
        :param ds_map: (numpy.ma)[Default: None] A masked array representing a map with discrete values.
                                       If None, then only the continuous part will be drawn.
        :param v_min: (float|None)[Default: 0] Minimum value the map can take.
                                               If None, it will be taken as cont_map.min().
        :param v_max: (float|None)[Default: 1] Maximum value the map can take.
                                               If None, it will be taken as cont_map.max().
        :param occupancy_map: (bool)[Default: True] If True, then 'Free' and 'Occ' will be appended to the first and
                                                    last tick labels respectively if they were not defined as None.

        :return: A tuple of a Matplotlib figure and axes objects
        """

        fig, ax = self._draw_plot(cont_map, ds_map, v_min=v_min, v_max=v_max, occupancy_map=occupancy_map)

        fig.show()

        return fig, ax

    def plot_save(self, path, cont_map, ds_map=None,
                  ds_list=None, v_min=0, v_max=1, occupancy_map=True, resolution=300):
        """
        Create and save without displaying a figure, draw the discrete map and color bar if not None, and then the
        continuous map and color bar.

        :param path: (string) Path where the resulting image is to be saved.
        :param cont_map: (numpy.ma) A masked array representing a map with continuous values.
        :param ds_map: (numpy.ma)[Default: None] A masked array representing a map with discrete values.
                                                 If None, then only the continuous part will be drawn.
        :param ds_list: (list)[Default: None] List of possible discrete state values.
                                              Used for drawing the discrete color bar. All states listed will be
                                              displayed in the bar, even if the image has no pixels with that value.
                                              E.g.: [DiscreteStates.UNIFORM, DiscreteStates.UNDEFINED]
        :param v_min: (float|None)[Default: 0] Minimum value the map can take.
                                               If None, it will be taken as cont_map.min().
        :param v_max: (float|None)[Default: 1] Maximum value the map can take.
                                               If None, it will be taken as cont_map.max().
        :param occupancy_map: (bool)[Default: True] If True, then 'Free' and 'Occ' will be appended to the first and
                                                    last tick labels respectively if they were not defined as None.
        :param resolution: (int)[Default: 300] Resolution of the saved image in DPI.

        :return: None
        """

        fig, ax = self._draw_plot(cont_map, ds_map,
                                  ds_list=ds_list, v_min=v_min, v_max=v_max, occupancy_map=occupancy_map)

        plt.savefig(path, bbox_inches='tight', dpi=resolution)

        plt.close(fig)


if __name__ == '__main__':
    """
    Test and sample code
    """

    import numpy.ma as ma
    import os.path

    hits = np.array([[0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 1, 1, 2, 0],
                     [0, 2, 1, 3, 0],
                     [0, 0, 0, 0, 1]])

    visits = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 9, 3, 0],
                       [0, 2, 2, 4, 0],
                       [0, 0, 0, 0, 8]])

    undef_mask = (visits == 0)
    alpha = ma.masked_array(hits, dtype=np.float)
    alpha[undef_mask] = ma.masked

    means = ma.divide(alpha, visits)

    means_ds = ma.zeros(means.shape)
    means_ds[undef_mask] = DiSt.UNDEFINED.value
    means_ds[~undef_mask] = ma.masked

    worldmap_extent = [150.4, 183.0, 0, 24.5]
    test_ds_list = [DiSt.UNDEFINED, DiSt.UNIFORM, DiSt.BIMODAL]

    test_v_min = 0
    test_v_max = 1

    test_occ = True

    # Create Colorizer Object
    mean_colorizer = MapColorizer()
    mean_colorizer.set_wm_extent(worldmap_extent)
    mean_colorizer.set_disc_state_list(test_ds_list)
    mean_colorizer.set_cb_tick_count(4)
    mean_colorizer.set_aspect_ratio('equal')
    mean_colorizer.set_cb_orientation('horizontal')

    # Save map to image file
    test_path = os.path.expanduser("~/Desktop/colorizer_test.svg")
    mean_colorizer.plot_save(test_path, means, means_ds, v_min=test_v_min, v_max=test_v_max, occupancy_map=test_occ)

    # Plot map to interactive window
    mean_colorizer.plot(means, means_ds, v_min=test_v_min, v_max=test_v_max, occupancy_map=test_occ)

    # Colorize (i.e. generate an RGBa image) from a map
    rgba = mean_colorizer.colorize(means, means_ds)
    # and then display it to compare.
    plt.figure()
    plt.imshow(rgba)
    plt.show()

    print('end')

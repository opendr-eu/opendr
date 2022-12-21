"""
This module implements methods for plotting.
Modified based on:
https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks
"""

from os import path, makedirs
import numpy as np
from matplotlib import pyplot as plt


def plot(data, title='Figure', legends=None, axis_x=None, axis_y=None, file_path=None, file_name=None,
         figure_size=(16, 9), has_grid=True, limits_axis_y=None, upper_lower_data=None, limits_axis_x=None,
         verbose=True):
    """
    Plot a graph from a list of x and y values.

    :param data: List of x and y values.
    :param title: Title of the graph (String).
    :param legends: List of legend (String) for each function.
    :param axis_x: Label (String) for the x-axis.
    :param axis_y: Label (String) for the y-axis.
    :param file_path: File path to save the Figure. If this variable is None the graph is shown to the user, otherwise,
                        the graph is saved only.
    :param file_name: File name to save the Figure. If this variable is None the graph is shown to the user, otherwise,
                        the graph is saved only.
    :param figure_size: Tuple containing the figure size (width, height).
    :param has_grid: Flag for a grid background.
    :param limits_axis_y: Tuple containing the limits (min, max, step) to the axis y.
    :param upper_lower_data: Tuple containing the upper and lower error limits for each data array.
    :param limits_axis_x: Tuple containing the limits (min, max, step) to the axis x.
    """

    plots = []
    colors = ['steelblue', 'indianred', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon',
              'blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'gray', 'sienna',
              'tan', 'plum', 'steelblue', 'lavenderblush', 'pink', 'navajowhite', 'darkorange',
              'darkslateblue', 'blueviolet', 'slategray', 'indianred', 'olive', 'darksalmon']

    plt.rcParams['figure.figsize'] = figure_size
    plt.title(title)
    plt.grid(has_grid)

    if not (axis_x is None):
        plt.xlabel(axis_x)
    if not (axis_y is None):
        plt.ylabel(axis_y)

    for d in range(len(data)):
        current_fig, = plt.plot(data[d][0], data[d][1], color=colors[d])
        if not (upper_lower_data is None):
            plt.fill_between(data[d][0], np.array(upper_lower_data[d][0], dtype=float),
                             np.array(upper_lower_data[d][1], dtype=float),
                             where=np.array(upper_lower_data[d][0], dtype=float) > np.array(upper_lower_data[d][1],
                                                                                            dtype=float), alpha=0.5,
                             interpolate=True)

        plots.append(current_fig)

    if not (legends is None):
        plt.legend(plots, legends)

    if not (limits_axis_y is None):
        plt.ylim(limits_axis_y[:2])
        plt.yticks(np.arange(limits_axis_y[0], limits_axis_y[1] + limits_axis_y[2], limits_axis_y[2]))

    if not (limits_axis_x is None):
        plt.xlim(limits_axis_x[:2])
        plt.xticks(np.arange(limits_axis_x[0], limits_axis_x[1] + limits_axis_x[2], limits_axis_x[2]))

    if (file_name is None) or (file_path is None):
        plt.show()
    else:
        full_path = path.join(file_path, file_name)
        if not path.isdir(file_path):
            makedirs(file_path)
        plt.savefig(full_path, format='svg')
        plt.close()
        if verbose:
            print('Figure saved at %s successfully.' % full_path)

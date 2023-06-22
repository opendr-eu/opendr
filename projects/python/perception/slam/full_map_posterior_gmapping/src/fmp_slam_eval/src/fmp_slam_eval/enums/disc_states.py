
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

from enum import Enum


class DiscreteStates(Enum):
    """
    Enum for possible discrete states such as undefined (division by 0), Uniform Distribution, etc.
    """

    #   KEY   | Val |   Text     |    Color        |  Label
    UNDEFINED = 1, 'undefined', 'lightsteelblue', 'Undef.'
    UNIFORM = 2, 'uniform', 'violet', 'Unif. (0, 1)'
    BIMODAL = 3, 'bimodal', 'mediumpurple', 'BiMod. {0, 1}'
    ZERO = 4, 'zero', 'white', '0'

    def __new__(cls, v, text, color, label):
        """
        Method for creating new Enum objects. Overridden to add attributes to each possible value.

        :param v: (int) Numerical value to be assigned to the enum. Also to be used as the value for the map cells.
        :param text: (string) Unused text to better describe the enum.
        :param color: (string|matplotlib color) Color to be assigned to this value.
        :param label: (string) Text to be displayed in the legend color bar.
        """

        obj = object.__new__(cls)
        obj._value_ = v
        obj.text = text
        obj.color = color
        obj.label = label
        return obj

    @classmethod
    def list_all(cls):
        """
        List all possible discrete states.

        :return: (list) List of all defined discrete states.
        """
        ds_list = [s for s in cls]
        return ds_list

    @classmethod
    def sort_ds_list(cls, ds_list):
        """
        Orders a given list of discrete states by their numerical value.

        :param ds_list: (list) List of Discrete State values.

        :return: (list) List of Discrete States ordered by value.
        """

        ordered_ds_list = sorted(ds_list, key=lambda x: x.value)
        return ordered_ds_list

    @classmethod
    def get_colors(cls, ds_list):
        """
        Get the color values for a given list of discrete states. The list is first ordered to maintain consistency.

        :param ds_list: (list) List of Discrete State values.

        :return: (list) List of colors corresponding to the (ordered) given states.
        """

        ds_list = cls.sort_ds_list(ds_list)
        return [state.color for state in ds_list]

    @classmethod
    def get_values(cls, ds_list):
        """
        Get the numeric values for a given list of discrete states. The list is first ordered to maintain consistency.

        :param ds_list: (list) List of Discrete State values.

        :return: (list) List of numeric values corresponding to the (ordered) given states.
        """

        ds_list = cls.sort_ds_list(ds_list)
        return [state.value for state in ds_list]

    @classmethod
    def get_labels(cls, ds_list):
        """
        Get the text labels for a given list of discrete states. The list is first ordered to maintain consistency.

        :param ds_list: (list) List of Discrete State values.

        :return: (list) List of string labels corresponding to the (ordered) given states.
        """
        ds_list = cls.sort_ds_list(ds_list)
        return [state.label for state in ds_list]


if __name__ == "__main__":
    """
    Test and sample code
    """

    obj_type = type(DiscreteStates.UNIFORM)
    print('Type Test:', obj_type)

    value = DiscreteStates.UNIFORM.value
    print('Value Test:', value)

    value = DiscreteStates.ZERO.value
    print('Value Test:', value)

    test_all_list = DiscreteStates.list_all()
    print('All Test:', test_all_list)

    test_ds_list = [DiscreteStates.UNIFORM, DiscreteStates.UNDEFINED, DiscreteStates.ZERO]
    print('DS List Test:', test_ds_list)

    ordered_test_ds_list = DiscreteStates.sort_ds_list(test_ds_list)
    print('Ordered DSList Test:', ordered_test_ds_list)

    colors = DiscreteStates.get_colors(test_ds_list)
    print('Get Colors Test:', colors)

    labels = DiscreteStates.get_labels(test_ds_list)
    print('Get Labels Test:', labels)

    values = DiscreteStates.get_values(test_ds_list)
    print('Get Values Test:', values)

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

import fire
from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_wrap,
    theme,
    element_text,
)
from pandas import DataFrame, Categorical
from plotnine.scales.limits import ylim


def plot_realtime():

    frame = DataFrame(
        {
            "Data FPS": [
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                10,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
                20,
            ],
            "Device": [
                "1080Ti",
                "1080Ti",
                "1080Ti",
                "Xavier",
                "Xavier",
                "Xavier",
                "TX2",
                "TX2",
                "TX2",
                "1080Ti",
                "1080Ti",
                "1080Ti",
                "Xavier",
                "Xavier",
                "Xavier",
                "TX2",
                "TX2",
                "TX2",
            ],
            "Method": [
                "P2B",
                "PTT",
                "VPIT",
                "P2B",
                "PTT",
                "VPIT",
                "P2B",
                "PTT",
                "VPIT",
                "P2B",
                "PTT",
                "VPIT",
                "P2B",
                "PTT",
                "VPIT",
                "P2B",
                "PTT",
                "VPIT",
            ],
            "Success": [
                56.20,
                67.80,
                50.49,
                36.50,
                63.60,
                50.49,
                21.90,
                29.50,
                50.31,
                56.20,
                67.80,
                50.49,
                16.70,
                26.50,
                47.70,
                10.90,
                17.90,
                38.96,
            ],
        }
    )

    frame["Device"] = Categorical(frame["Device"], ["1080Ti", "Xavier", "TX2"])

    def plot_frame(frame, output_file_name):

        plot = (
            ggplot(frame) +
            aes(x="Device", y="Success") +
            facet_wrap(["Data FPS"], nrow=2, labeller="label_both") +
            geom_point(
                aes(colour="Method"),
            ) +
            geom_line(aes(colour="Method", group="Method"))
        )

        plot.save("plots/" + output_file_name + ".png", dpi=600)

    plot_frame(frame, "realtime")


def plot_ablation(lines_count):

    frames = [
        {
            "x": "Blocks",
            "data": DataFrame(
                {
                    "Blocks": Categorical([1, 2, 3], [1, 2, 3]),
                    "Success": [42.75, 37.69, 26.54],
                }
            ),
        },
        {
            "x": "Target/search region size",
            "data": DataFrame(
                {
                    "Target/search region size": Categorical(
                        ["Original", "Upscaled"], ["Original", "Upscaled"]
                    ),
                    "Success": [42.75, 23.58],
                }
            ),
        },
        {
            "x": "Positive label radius",
            "data": DataFrame(
                {
                    "Positive label radius": [1, 2, 4, 8, 16],
                    "Success": [40.58, 42.75, 41.86, 40.64, 38.78],
                }
            ),
        },
        {
            "x": "Context amount",
            "data": DataFrame(
                {
                    "Context amount": [
                        -0.3,
                        -0.2,
                        -0.1,
                        0,
                        0.1,
                        0.2,
                        0.24,
                        0.25,
                        0.26,
                        0.27,
                        0.3,
                        0.31,
                        0.32,
                    ],
                    "Success": [
                        28.62,
                        28.43,
                        38.76,
                        31.30,
                        33.68,
                        40.64,
                        41.63,
                        41.74,
                        42.75,
                        39.58,
                        40.05,
                        41.23,
                        40.89,
                    ],
                }
            ),
        },
        # {
        #     "x": "Target feature merge scale",
        #     "data": DataFrame(
        #         {
        #             "Target feature merge scale": Categorical(
        #                 ["0", "0.005", "0.01"], ["0", "0.005", "0.01"]
        #             ),
        #             "Success": [42.02, 42.75, 41.51],
        #         }
        #     ),
        # },
        # {
        #     "x": "Window influence",
        #     "data": DataFrame(
        #         {
        #             "Window influence": [0.35, 0.45, 0.5, 0.65, 0.85, 0.9],
        #             "Success": [40.83, 41.86, 41.15, 42.58, 42.75, 42.47],
        #         }
        #     ),
        # },
        {
            "x": "Score upscale",
            "data": DataFrame(
                {
                    "Score upscale": [1, 2, 4, 8, 16, 32],
                    "Success": [40.40, 40.65, 39.15, 42.75, 42.49, 40.66],
                }
            ),
        },
        # {
        #     "x": "Search region scale",
        #     "data": DataFrame(
        #         {
        #             "Search region scale": [1.5, 2],
        #             "Success": [42.75, 40.04],
        #         }
        #     ),
        # },
        {
            "x": "Position extrapolation",
            "data": DataFrame(
                {
                    "Position extrapolation": Categorical(
                        ["None", "Linear"], ["None", "Linear"]
                    ),
                    "Success": [32.80, 42.75],
                }
            ),
        },
        {
            "x": "Offset interpolation",
            "data": DataFrame(
                {
                    "Offset interpolation": [0.2, 0.3, 0.4, 0.75, 1],
                    "Success": [36.16, 42.75, 39.75, 38.94, 34.60],
                }
            ),
        },
        {
            "x": "Training steps",
            "data": DataFrame(
                {
                    "Training steps": Categorical(
                        [
                            "Pretrained",
                            "8k",
                            "16k",
                            "32k",
                            "64k",
                            "86k",
                            "128k",
                        ],
                        [
                            "Pretrained",
                            "8k",
                            "16k",
                            "32k",
                            "64k",
                            "86k",
                            "128k",
                        ],
                    ),
                    "Success": [22.18, 37.55, 38.71, 40.86, 42.75, 40.40, 39.77],
                }
            ),
        },
    ]

    def plot_frames(frames, output_file_name):

        for i, frame in enumerate(frames):
            plot = (
                ggplot(frame["data"]) +
                aes(x=frame["x"], y="Success") +
                ylim(22, 43) +
                geom_point(
                    aes(),
                ) +
                geom_line(group=frame["x"]) +
                theme(
                    axis_title=element_text(size=20), axis_text=element_text(size=14)
                )
            )

            plot.save("plots/" + output_file_name + str(i) + ".png", dpi=600)

    def combine_plots(input_file_name, output_file_name, lines_count=2):

        import cv2
        import numpy as np
        import math

        lines = []

        images_in_line = math.ceil(len(frames) / lines_count)

        i = 0

        for q in range(lines_count):

            lines.append(None)

            for p in range(
                images_in_line
                if q < lines_count - 1
                else (len(frames) - images_in_line * (lines_count - 1))
            ):
                image = cv2.imread("plots/" + input_file_name + str(i) + ".png")

                if lines[-1] is None:
                    lines[-1] = image
                else:
                    lines[-1] = np.concatenate((lines[-1], image), axis=1)

                i += 1

            if q == lines_count - 1:
                for p in range(
                    images_in_line - (len(frames) - images_in_line * (lines_count - 1))
                ):
                    image_zero = np.ones_like(image) * 255
                    lines[-1] = np.concatenate((lines[-1], image_zero), axis=1)

        # last = cv2.resize(cv2.imread("plots/" + input_file_name + str(len(frames) - 1) + ".png"), None, fx=2, fy=2)

        all = np.concatenate(lines, axis=0)
        # all = np.concatenate((all, last), axis=1)

        cv2.imwrite("plots/" + output_file_name + ".png", all)
        cv2.imwrite(
            "plots/" + output_file_name + "_small.png",
            cv2.resize(all, None, fx=0.5, fy=0.5),
        )

    plot_frames(frames, "ablation", lines_count=lines_count)
    combine_plots("ablation", "all_ablation")


if __name__ == "__main__":

    fire.Fire()

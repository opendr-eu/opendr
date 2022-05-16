from plotnine import (
    ggplot,
    aes,
    geom_line,
    geom_point,
    facet_grid,
    facet_wrap,
    scale_y_continuous,
    geom_hline,
    position_dodge,
    geom_errorbar,
    theme,
    element_text,
)
from pandas import DataFrame, Categorical
from plotnine.scales.limits import ylim


frame = DataFrame({
    "Data FPS": [10, 10, 10, 10, 10, 10, 10, 10, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20],
    "Device": [
        "1080Ti", "1080Ti", "1080Ti", "Xavier", "Xavier", "Xavier", "TX2", "TX2", "TX2",
        "1080Ti", "1080Ti", "1080Ti", "Xavier", "Xavier", "Xavier", "TX2", "TX2", "TX2"
    ],
    "Method": [
        "P2B", "PTT", "VPIT", "P2B", "PTT", "VPIT", "P2B", "PTT", "VPIT",
        "P2B", "PTT", "VPIT", "P2B", "PTT", "VPIT", "P2B", "PTT", "VPIT"
    ],
    "Success": [
        56.20, 67.80, 50.49, 36.50, 63.60, 50.49, 21.90, 29.50, 50.31,
        56.20, 67.80, 50.49, 16.70, 26.50, 47.70, 10.90, 17.90, 38.96
    ],
})

frame["Device"] = Categorical(frame["Device"], ["1080Ti", "Xavier", "TX2"])


def plot_frame(frame, output_file_name):

    plot = (
        ggplot(frame)
        + aes(x="Device", y="Success")
        + facet_wrap(["Data FPS"], nrow=2, labeller="label_both")
        + geom_point(
            aes(colour="Method"),
            # size=3,
            # position=position_dodge(width=0.8),
            # stroke=0.2,
        )
        + geom_line(aes(colour="Method", group="Method"))
    )

    # if len(params["facet"]) > 2:
    #     plot = plot + theme(strip_text_x=element_text(size=5))

    plot.save("plots/" + output_file_name + ".png", dpi=600)


plot_frame(frame, "realtime")

import cv2
import numpy as np
import math
from metrics import MinMetric, MaxMetric
from PIL import Image, ImageDraw


def draw_point_cloud_bev(
    point_cloud, predictions=None, scale=10,
    xs=[-90, 90], ys=[-90, 90]
):
    x_min = MinMetric()
    y_min = MinMetric()
    x_max = MaxMetric()
    y_max = MaxMetric()

    # x_min.update(0)
    # x_max.update(90)
    # y_min.update(-40)
    # y_max.update(40)

    x_min.update(xs[0])
    x_max.update(xs[1])
    y_min.update(ys[0])
    y_max.update(ys[1])

    point_cloud = point_cloud[point_cloud[:, 0] > x_min.get()]
    point_cloud = point_cloud[point_cloud[:, 0] < x_max.get()]
    point_cloud = point_cloud[point_cloud[:, 1] > y_min.get()]
    point_cloud = point_cloud[point_cloud[:, 1] < y_max.get()]

    x_size = x_max.get() - x_min.get()
    y_size = y_max.get() - y_min.get()

    image_size_x = int(x_size * scale + 1)
    image_size_y = int(y_size * scale + 1)

    point_cloud_x = (
        image_size_x - 1 - (point_cloud[:, 0] - x_min.get()) * scale
    ).astype(np.int32)
    point_cloud_y = ((point_cloud[:, 1] - y_min.get()) * scale).astype(
        np.int32
    )

    colors = np.array([255, 255, 255], dtype=np.uint8)

    color_image = np.zeros([image_size_x, image_size_y, 3], dtype=np.uint8)

    def rotate_rectangle(x, y, half_size_x, half_size_y, angle):
        def distance(ax, ay, bx, by):
            return math.sqrt((by - ay) ** 2 + (bx - ax) ** 2)

        # rotates point `A` about point `B` by `angle` radians clockwise.
        def rotated_about(ax, ay, bx, by, angle):
            radius = distance(ax, ay, bx, by)
            angle += math.atan2(ay - by, ax - bx)
            return (
                round(bx + radius * math.cos(angle)),
                round(by + radius * math.sin(angle)),
            )

        vertices = (
            (x + half_size_x, y + half_size_y),
            (x + half_size_x, y - half_size_y),
            (x - half_size_x, y - half_size_y),
            (x - half_size_x, y + half_size_y),
        )

        result = [rotated_about(vx, vy, x, y, angle) for vx, vy in vertices]

        return [(y, x) for (x, y) in result]

    pil_image = Image.new(
        "RGB", (image_size_y + 1, image_size_x + 1), color="black"
    )
    pil_draw = ImageDraw.Draw(pil_image)

    for box in predictions:

        x_3d, y_3d = box.location[:2]
        size = box.dimensions
        rotation = box.rotation_y
        x_bev = (image_size_x - 1 - (x_3d - x_min.get()) * scale).astype(
            np.int32
        )
        y_bev = ((y_3d - y_min.get()) * scale).astype(np.int32)

        half_size_x, half_size_y = (size[:2] * scale / 2).astype(np.int32)

        pil_draw.polygon(
            rotate_rectangle(
                x_bev,
                y_bev,
                half_size_x,
                half_size_y,
                rotation,  # + math.pi / 2
            ),
            fill=(192, 102, 50),
            outline=(255, 0, 255),
        )

    color_image = np.array(pil_image)
    color_image[point_cloud_x, point_cloud_y] = colors

    return color_image

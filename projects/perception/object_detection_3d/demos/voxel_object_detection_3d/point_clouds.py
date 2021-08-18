import cv2
import numpy as np
import math
from metrics import MinMetric, MaxMetric

def draw_point_cloud_bev(
    point_cloud,
    boxes=[],
    scale=5,
):
    x_min = MinMetric()
    y_min = MinMetric()
    d_min = MinMetric()
    z_min = MinMetric()
    x_max = MaxMetric()
    y_max = MaxMetric()
    z_max = MaxMetric()
    d_max = MaxMetric()

    x_min.update(-90)
    x_max.update(90)

    y_min.update(-90)
    y_max.update(90)

    # for i in range(len(point_cloud)):
    #     x, y, z = point_cloud[i, :3]

    #     d = math.sqrt(x*x + y*y + z*z)

    #     x_min.update(x)
    #     x_max.update(x)
    #     y_min.update(y)
    #     y_max.update(y)
    #     z_min.update(z)
    #     z_max.update(z)

    x_size = x_max.get() - x_min.get()
    y_size = y_max.get() - y_min.get()

    image_size_x = int(x_size * scale + 1)
    image_size_y = int(y_size * scale + 1)

    point_cloud_x = (
        image_size_x - 1 - (point_cloud[:, 0] - x_min.get()) * scale
    ).astype(np.int32)
    point_cloud_y = ((point_cloud[:, 1] - y_min.get()) * scale).astype(np.int32)
    # point_cloud_d = 1 - 0.2 * (point_cloud[:, 2] - d_min.get()) / (
    #     d_max.get() - d_min.get()
    # )
    # colors = np.stack(
    #     [
    #         255, #  point_cloud_d,
    #         255, #  point_cloud_d * (1 - point_cloud[:, 3]),
    #         255, #  point_cloud_d * (1 - point_cloud[:, 3]),
    #     ],
    #     axis=-1,
    # ).astype(np.uint8)

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

    for (x, y, z, w, h, d, rotation) in boxes:
        # center_3d = image[y, x][:3] + offset
        # x_3d, y_3d = center_3d[:2]
        # x_bev = (image_size_x - 1 - (x_3d - x_min.get()) * scale).astype(
        #     np.int32
        # )
        # y_bev = ((y_3d - y_min.get()) * scale).astype(np.int32)

        # half_size_x, half_size_y = (size[:2] * scale / 2).astype(np.int32)

        # pil_draw.polygon(
        #     rotate_rectangle(
        #         x_bev, y_bev, half_size_x, half_size_y, rotation + math.pi / 2
        #     ),
        #     fill=(192, 102, 50),
        #     outline=(255, 0, 255),
        # )
        pass

    # color_image = (np.array(pil_image) / 255.0).astype(np.float32)
    color_image[point_cloud_x, point_cloud_y] = colors

    return color_image
from .deep_sort import DeepSort


__all__ = ["DeepSort", "build_tracker"]


def build_tracker(
    max_dist,
    min_confidence,
    nms_max_overlap,
    max_iou_distance,
    max_age,
    n_init,
    nn_budget,
    device
):
    return DeepSort(
        max_dist=max_dist,
        min_confidence=min_confidence,
        nms_max_overlap=nms_max_overlap,
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
        nn_budget=nn_budget,
        device=device,
    )

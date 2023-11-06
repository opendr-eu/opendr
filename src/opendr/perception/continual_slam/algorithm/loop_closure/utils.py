from typing import Optional

import matplotlib.pyplot as plt


def plot_image_matches(
    image_0,
    image_1,
    image_id_0: Optional[int] = None,
    image_id_1: Optional[int] = None,
    cosine_similarity: Optional[float] = None,
    save_figure: bool = True,
) -> None:
    fig = plt.figure()
    plt.subplot(211)
    plt.imshow(image_0)
    plt.axis('off')
    if image_id_0 is not None:
        plt.title(image_id_0)
    plt.subplot(212)
    plt.imshow(image_1)
    plt.axis('off')
    if image_id_1 is not None:
        plt.title(image_id_1)
    if cosine_similarity is not None:
        plt.suptitle(f'cos_sim = {cosine_similarity}')
    if save_figure:
        assert image_id_0 is not None and image_id_1 is not None
        plt.savefig(f'./figures/sequence_08/matches/{image_id_0:04}_{image_id_1:04}.png')
    else:
        plt.show()
    plt.close(fig)

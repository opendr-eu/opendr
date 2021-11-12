from ..dataset.retinaface import retinaface


def load_gt_roidb(dataset_name, image_set_name, root_path, dataset_path,
                  flip=False, verbose=False):
    """ load ground truth roidb """
    imdb = retinaface(image_set_name, root_path, dataset_path)
    roidb = imdb.gt_roidb()
    if verbose:
        print('roidb size', len(roidb))
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    if verbose:
        print('flipped roidb size', len(roidb))
    return roidb

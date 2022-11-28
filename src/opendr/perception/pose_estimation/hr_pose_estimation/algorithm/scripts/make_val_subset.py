import json
import random


def make_val_subset(labels, output_path="val_subset.json", num_images=250):
    """
    :param labels: path to json with keypoints val labels
    :param output_path: name of output file with subset of val labels, defaults to "val_subset.json"
    :param num_images: number of images in subset, defaults to 250
    """
    with open(labels, 'r') as f:
        data = json.load(f)

    random.seed(0)
    total_val_images = 5000
    idxs = list(range(total_val_images))
    random.shuffle(idxs)

    images_by_id = {}
    for idx in idxs[:num_images]:
        images_by_id[data['images'][idx]['id']] = data['images'][idx]

    annotations_by_image_id = {}
    for annotation in data['annotations']:
        if annotation['image_id'] in images_by_id:
            if not annotation['image_id'] in annotations_by_image_id:
                annotations_by_image_id[annotation['image_id']] = []
            annotations_by_image_id[annotation['image_id']].append(annotation)

    subset = {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': [],
        'annotations': [],
        'categories': data['categories']
    }
    for image_id, image in images_by_id.items():
        subset['images'].append(image)
        if image_id in annotations_by_image_id:  # image has at least 1 annotation
            subset['annotations'].extend(annotations_by_image_id[image_id])

    with open(output_path, 'w') as f:
        json.dump(subset, f, indent=4)

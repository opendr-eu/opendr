'''
Modifications Copyright 2023 - present, OpenDR European Project

This code is modified from
https://github.com/hukenovs/hagrid/blob/master/converters/hagrid_to_coco.py
under public license at https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf
'''

import json
import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGES = (".jpeg", ".jpg", ".jp2", ".png", ".tiff", ".jfif", ".bmp", ".webp", ".heic")

tqdm.pandas()

logging.getLogger().setLevel(logging.INFO)


def get_area(bboxes: list) -> list:
    """
    Get area of bboxes
    Parameters
    ----------
    bboxes: list
        list of bboxes

    Returns
    -------
    list
    """
    bboxes = np.array(bboxes)
    area = bboxes[:, 2] * bboxes[:, 3]
    return area


def get_w_h(img_path: str) -> Tuple[int, int]:
    """
    Get width and height of image
    Parameters
    ----------
    img_path: str
        path to image

    Returns
    -------
    Tuple[int, int]
    """
    img = Image.open(img_path)
    img_w, img_h = img.size
    return img_w, img_h


def get_abs_bboxes(bboxes: list, im_size: tuple) -> list:
    """
    Get absolute bboxes in format [xmin, ymin, w, h]
    Parameters
    ----------
    bboxes: list
        list of bboxes
    im_size: tuple
        image size

    Returns
    -------
    list
    """
    width, height = im_size
    bboxes_out = []
    for box in bboxes:
        x1, y1, w, h = box
        bbox_abs = [x1 * width, y1 * height, w * width, h * height]
        bboxes_out.append(bbox_abs)
    return bboxes_out


def get_poly(bboxes: list) -> list:
    """
    Get polygon from bboxes
    Parameters
    ----------
    bboxes: list
        list of bboxes

    Returns
    -------
    list
    """
    poly = []
    for box in bboxes:
        xmin, ymin, w, h = box
        xmax = xmin + w
        ymax = ymin + h
        poly.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])

    return poly


def get_files_from_dir(pth: str, extns: Tuple) -> list:
    """
    Get files from directory
    Parameters
    ----------
    pth: str
        path to directory
    extns: Tuple
        extensions of files

    Returns
    -------
    list
    """
    if not os.path.exists(pth):
        logging.error(f"Dataset directory doesn't exist {pth}")
        return []
    files = [f for f in os.listdir(pth) if f.endswith(extns)]
    return files


def get_dataframe(dataset_annotations, dataset_folder, targets, phase):
    annotations_all = None
    exists_images = []

    for target in tqdm(targets):
        ps_phase = 'train_val' if (phase == 'train' or phase == 'val') else phase
        target_json = os.path.join(dataset_annotations, f"ann_{ps_phase}", f"{target}.json")
        print('target json: ', target_json)
        if os.path.exists(target_json):
            json_annotation = json.load(open(os.path.join(target_json)))

            json_annotation = [
                dict(annotation, **{"name": f"{name}.jpg"})
                for name, annotation in zip(json_annotation, json_annotation.values())
            ]

            annotation = pd.DataFrame(json_annotation)

            annotation["target"] = target
            annotations_all = pd.concat([annotations_all, annotation], ignore_index=True)
            exists_images.extend(get_files_from_dir(os.path.join(dataset_folder, phase, target), IMAGES))
        else:
            logging.warning(f"Database for {phase}/{target} not found")

    annotations_all["exists"] = annotations_all["name"].isin(exists_images)
    annotations = annotations_all[annotations_all["exists"]]

    return annotations


def convert_to_coco(out='./hagrid_coco_format', dataset_folder='./data/', dataset_annotations='./data/annotations/'):
    targets = ['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace',
               'rock', 'stop', 'stop_inverted', 'three', 'two_up', 'two_up_inverted',
               'three2', 'peace_inverted', 'no_gesture']
    labels = {label: num for (label, num) in zip(targets, range(len(targets)))}
    if not os.path.exists(out):
        os.makedirs(out)
    phases = ['train', 'val', 'test']
    for phase in phases:

        logging.info(f"Run convert {phase}")
        logging.info("Create Dataframe")
        annotations = get_dataframe(dataset_annotations, dataset_folder, targets, phase)

        logging.info("Create image_path")
        annotations["image_path"] = annotations.progress_apply(
            lambda row: os.path.join(dataset_folder, phase, row["target"], row["name"]), axis=1
        )

        logging.info("Create width, height")
        w_h = annotations["image_path"].progress_apply(lambda x: get_w_h(x))
        annotations["width"] = np.array(w_h.to_list())[:, 0]
        annotations["height"] = np.array(w_h.to_list())[:, 1]

        logging.info("Create id")
        annotations["id"] = annotations.index

        logging.info("Create abs_bboxes")
        annotations["abs_bboxes"] = annotations.progress_apply(
            lambda row: get_abs_bboxes(row["bboxes"], (row["width"], row["height"])), axis=1
        )
        logging.info("Create area")
        annotations["area"] = annotations["abs_bboxes"].progress_apply(lambda bboxes: get_area(bboxes))
        logging.info("Create segmentation")
        annotations["segmentation"] = annotations["abs_bboxes"].progress_apply(lambda bboxes: get_poly(bboxes))
        logging.info("Create category_id")
        annotations["category_id"] = annotations["labels"].progress_apply(lambda x: [labels[label] for label in x])

        categories = [{"supercategory": "none", "name": k, "id": v} for k, v in labels.items()]
        logging.info(f"Save to {phase}.json")
        res_file = {"categories": categories, "images": [], "annotations": []}
        annot_count = 0
        for index, row in tqdm(annotations.iterrows()):
            img_elem = {"file_name": row["image_path"], "height": row["height"], "width": row["width"], "id": row["id"]}

            res_file["images"].append(img_elem)

            num_boxes = len(row["bboxes"])
            for i in range(num_boxes):
                annot_elem = {
                    "id": annot_count,
                    "bbox": row["abs_bboxes"][i],
                    "segmentation": [row["segmentation"][i]],
                    "image_id": row["id"],
                    "category_id": row["category_id"][i],
                    "iscrowd": 0,
                    "area": row["area"][i],
                }
                res_file["annotations"].append(annot_elem)
                annot_count += 1

        with open(f"{out}/{phase}.json", "w") as f:
            json_str = json.dumps(res_file)
            f.write(json_str)

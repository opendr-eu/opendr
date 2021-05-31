from PIL import Image
from .detector import detect_faces
from .align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm


def face_align(data=None, dest='None', crop_size=112):
    source_root = data
    dest_root = dest
    crop_size = crop_size
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    cwd = os.getcwd()
    os.chdir(source_root)
    os.system("find . -name '*.DS_Store' -type f -delete")
    os.chdir(cwd)

    if not os.path.isdir(dest_root):
        os.makedirs(dest_root)

    for subfolder in tqdm(os.listdir(source_root)):
        if not os.path.isdir(os.path.join(dest_root, subfolder)):
            os.mkdir(os.path.join(dest_root, subfolder))
        for image_name in os.listdir(os.path.join(source_root, subfolder)):
            img = Image.open(os.path.join(source_root, subfolder, image_name))
            try:
                _, landmarks = detect_faces(img)
            except Exception:
                print("{} is discarded due to exception!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            if len(landmarks) == 0:
                print("{} is discarded due to non-detected landmarks!".format(os.path.join(source_root, subfolder, image_name)))
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            img_warped = Image.fromarray(warped_face)
            if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']:
                image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
            img_warped.save(os.path.join(dest_root, subfolder, image_name))

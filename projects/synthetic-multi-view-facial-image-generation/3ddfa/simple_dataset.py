from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
from utils.inference import crop_img, parse_roi_box_from_landmark


def cv2_loader(img_str):
    img_array = np.frombuffer(img_str, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

class McDataset(Dataset):
    def __init__(self, img_list, landmarks, std_size=120, transform=None):

        self.img_list = img_list
        self.landmarks = landmarks
        self.transform = transform
        self.std_size = std_size
        assert len(self.img_list) == len(self.landmarks)
        self.num = len(self.img_list)

        self.initialized = False

    def __len__(self):
        return self.num

    def __getitem__(self, idx):

        filename = self.img_list[idx]
        ori_img = cv2.imread(filename)

        landmark = self.landmarks[idx]

        # preprocess img
        roi_box = parse_roi_box_from_landmark(landmark.T)
        img = crop_img(ori_img, roi_box)
        img = cv2.resize(img, dsize=(self.std_size, self.std_size), interpolation=cv2.INTER_LINEAR)
        if self.transform is not None:
            img = self.transform(img)

        return img, ori_img, filename, np.array(roi_box)

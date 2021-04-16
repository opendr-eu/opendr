from PIL import Image
import requests
from perception.object_detection_2d.pixel_object_detection_2d_learner import PixelObjectDetection2DLearner
from engine.datasets import ExternalDataset
from util.plot_utils import plot_logs
from pathlib import Path


def main():
    learner = PixelObjectDetection2DLearner(iters=3)
    learner.download()
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    im = Image.open(requests.get(url, stream=True).raw)
    # im = Image.open("/home/jelle/Afbeeldingen/kat.jpg")
    learner.infer(im)
    # dataset = ExternalDataset("/home/jelle/Afbeeldingen/small_coco", "coco")
    # learner.fit(dataset, annotations_folder="", 
    #             train_annotations_file="instances_train2017_small.json", 
    #             train_images_folder="train_2017_small", logging_path="/tmp", 
    #             val_annotations_file="instances_train2017_small.json", 
    #             val_images_folder="train_2017_small")
    # log_directory = [Path("/tmp")]
    # fields_of_interest = (
    #     'loss',
    #     'mAP',
    #     )
    
    # plot_logs(log_directory,
    #           fields_of_interest)
    
    # fields_of_interest = (
    #     'loss_ce',
    #     'loss_bbox',
    #     'loss_giou',
    #     )
    
    # plot_logs(log_directory,
    #           fields_of_interest)
    
    # fields_of_interest = (
    #     'class_error',
    #     'cardinality_error_unscaled',
    #     )
    
    # plot_logs(log_directory,
    #           fields_of_interest)


if __name__ == '__main__':
    main()

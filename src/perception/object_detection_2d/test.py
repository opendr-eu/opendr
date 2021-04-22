from PIL import Image as im
import requests
from perception.object_detection_2d.pixel_object_detection_2d_learner import PixelObjectDetection2DLearner
from engine.data import Image
from engine.datasets import ExternalDataset
# from algorithm.util.plot_utils import plot_logs
from pathlib import Path
# from algorithm.util.plot_utils import plot_results

def main():
    learner = PixelObjectDetection2DLearner(iters=1)
    learner.download()
    # success = learner.save("/home/jelle")
    # learner.load("/home/jelle")
    # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    # image = im.open(requests.get(url, stream=True).raw)
    # image = Image(image)
    # image = Image.open("/home/jelle/Afbeeldingen/kat.jpg")
    # blist = learner.infer(image)
    # for box in blist:
    #     print(box.name)
    # learner.optimize()
    # learner.save("/home/jelle")
    dataset = ExternalDataset("/home/jelle/Afbeeldingen/small_coco", "coco")
    learner.fit(dataset, annotations_folder="", 
                train_annotations_file="instances_train2017_small.json", 
                train_images_folder="train_2017_small", logging_path="/tmp", 
                val_annotations_file="instances_train2017_small.json", 
                val_images_folder="train_2017_small", verbose=True)
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

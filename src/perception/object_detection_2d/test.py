from perception.object_detection_2d.pixel_object_detection_2d_learner import PixelObjectDetection2DLearner
from engine.datasets import ExternalDataset

def main():
    learner = PixelObjectDetection2DLearner(iters=10)
    learner.load()
    dataset = ExternalDataset("/home/jelle/Afbeeldingen/small_coco", "coco")
    learner.fit(dataset, annotations_folder="", train_annotations_file="instances_train2017_small.json", train_images_folder="train_2017_small")

if __name__ == '__main__':
    main()

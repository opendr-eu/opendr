from pathlib import Path
from opendr.perception.panoptic_segmentation import EfficientLpsLearner, SemanticKittiDataset


def train():
	train_data = SemanticKittiDataset("/home/arceyd/MasterThesis/dat/kitti/dataset", split="train")
	val_data = SemanticKittiDataset("/home/arceyd/MasterThesis/dat/kitti/dataset", split="valid")

	learner = EfficientLpsLearner(
		iters=2,
		batch_size=1,
		checkpoint_after_iter=2
	)

	train_stats = learner.fit(train_data, val_dataset=val_data,
				logging_path=str(Path(__file__).parent / 'work_dir'))

	learner.save(path=)

if __name__ == "__main__":

	train()

	print("hola")

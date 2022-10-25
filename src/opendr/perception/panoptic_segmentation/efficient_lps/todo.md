# TODOs
___
## Coding
- -[X] Add EfficientLPS submodule
- -[ ] Create Learner class in `src/opendr/perception/panoptic_segmentation/efficient_lps`
    - -[X] Implement & test download method.
    - -[X] Implement & test fit() method.
    - -[X] Implement & test eval() method.
    - -[X] Implement & test infer() method.
    - -[X] Implement & test visualize() method.
    - -[X] Docstrings and Type-Hints
    
### CONFLICTS:
- -[X] Check installation and dependency conflicts between mmdet2 and EfficientNet in EfficientLPS and EfficientPS.
### ROS
- -[X] Create ROS Node in `projects/opendr_ws/perception/scripts`
   - -[X] Docstrings and Type-Hints

### Demo 
- -[X] Create & test Demo script in `projects/perception/panoptic_segmentation/efficient_lps`
    - -[X] Docstrings and Type-Hints
    
### Datasets
- -[ ] Create Dataset Classes:
    - -[ ] NuScenes in `src/opendr/perception/panoptic_segmentation/nuscenes.py`
        - -[ ] Test data for training.
        - -[ ] Test data for inference.
        - -[ ] Implement & test evaluate() method.
        - -[ ] Implement & test iterator methods \_\_getitem\_\_() and \_\_len\_\_().
        - -[ ] Docstrings and Type-Hints
        
    - -[X] Kitti in `src/opendr/perception/panoptic_segmentation/semantic_kitti.py`
        - -[X] Test data for training.
        - -[X] Test data for inference.
        - -[X] Implement & test evaluate() method.
        - -[X] Implement & test iterator methods \_\_getitem\_\_() and \_\_len\_\_().
        - -[X] Docstrings and Type-Hints

    
### Tests
- -[X] Create Unit tests in `tests/sources/tools/perception/panoptic_segmentation/efficient_lps`
- -[ ] Add tests to be executed to some file

## Documentation
- -[X] Create main README in `docs/reference/efficient_lps`
    - -[X] Add link to readme in `docs/reference/index.md`
    

- -[X] Update functionality README in `src/opendr/perception/panoptic_segmentation/README.md`
    

- -[ ] Update dataset README in `src/opendr/perception/panoptic_segmentation/datasets/README.md`
    - -[X] For KITTI dataset
    - -[ ] For NuScenes dataset
    

- -[X] Add README for demo usage in `projects/perception/panoptic_segmentation/efficient_lps`

## Data
- -[X] Upload to OpenDR server:
    - -[ ] Pre-trained model weights (for unittest and inference)
        - -[X] SemanticKITTI pre-trained model weights.
        - -[ ] NuScenes pre-trained model weights.
    - -[X] Test data (for unittest)
    - -[X] Update download Links in EfficientLpsLearner Class 
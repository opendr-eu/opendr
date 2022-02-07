# TODOs
___
## Coding
- -[X] Add EfficientLPS submodule
- -[ ] Create Learner class in `src/opendr/perception/panoptic_segmentation/efficient_lps`
    - -[ ] Implement & test download method.
    - -[X] Implement & test fit() method.
    - -[X] Implement & test eval() method.
    - -[X] Implement & test infer() method.
    - -[X] Implement & test visualize() method.
    - -[X] Docstrings and Type-Hints
    
### CONFLICTS:
- -[ ] Check installation and dependency conflicts between MMDet and EfficientNet in EfficientLPS and EfficientPS.
### ROS
- -[ ] Create ROS Node in `projects/opendr_ws/perception/scripts`
   - -[ ] Docstrings and Type-Hints

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
        
    - -[ ] Kitti in `src/opendr/perception/panoptic_segmentation/semantic_kitti.py`
        - -[X] Test data for training.
        - -[X] Test data for inference.
        - -[X] Implement & test evaluate() method.
        - -[ ] Implement & test iterator methods \_\_getitem\_\_() and \_\_len\_\_().
        - -[ ] Docstrings and Type-Hints

    
### Tests
- -[ ] Create Unit tests in `tests/tools/perception/panoptic_segmentation/efficient_lps`
- -[ ] Add tests to be executed to some file

## Documentation
- -[ ] Create main README in `docs/reference/efficient_lps`
    - -[ ] Add link to readme in `docs/reference/index.md`
    
- -[ ] Update functionality README in `src/opendr/perception/panoptic_segmentation/README.md`
    

- -[ ] Update dataset README in `src/opendr/perception/panoptic_segmentation/datasets/README.md`
    - -[X] For KITTI dataset
    - -[ ] For NuScenes dataset
    

- -[X] Add README for demo usage in `projects/perception/panoptic_segmentation/efficient_lps`

## Data
- -[ ] Upload to OpenDR server:
    - -[ ] Pre-trained model weights (for unittest and inference)
    - -[ ] Test data (for unittest)
# OpenDR Contribution Checklist

This file provides an overview of things that each contributing partner should check before starting a PR. This list would be also useful for the review process, allowing code maintainers to ensure that minimum standards regarding the code quality and functionality are met.


## Components expected from a contribution
Each DL perception-based method contributed to OpenDR (WP3-4) is expected to include the following:

1. Implementation of the method
2. Documentation on how the method is used and description of each function provided by the implementation
3. Usage examples
4. ROS/ROS2 nodes
5. Pre-trained models
6. Optimization methods built-in (e.g., ONNX, TensorRT, OpenVINO, etc.) [probably optional, but highly recommended]
7. C Interface [optional, but highly recommended]


In each PR, please explicitly state which of these you are providing and which you are going to provide in the future, in order to avoid any comments regarding missing items.


NOTE: *These might be different for methods that concern other WPs (e.g., WP5-6).*

## Pre-submission checklist

Contributors: Please consider the following points before submitting a PR. 
Reviewers: Please check all these points for each PR.



1. The main functionality is provided in a Learner-based class.
2. The Learner implementation follows the function signatures provided in Appendix A.1 of D2.1.
3. infer(), train() and eval() work correctly with OpenDR data types (e.g., Dataset, Image, Target, etc.).
4. save() and load() functions are provided.
5. Data format specifications (Section 5.10.2 of D2.1) are followed for trained models.
6. Any binary models or data are removed from the contributed files. Binary models and/or data should go to the FTP server.
7. There is no *duplicate* functionality in the code that is provided elsewhere (e.g. pip package, other repo that can be linked, etc.). If so, this should not be re-included in the repository.
8. There is no leftover functionality that is not going to be used in the contributed code. If so, this should be removed to make the code easier to maintain.
9. Appropriate tests to ensure that my code runs correctly (at least one interface check that checks that model works correctly with OpenDR datatypes and at least one functionality check with a pre-trained model).
10. Documentation has been included.
11. At least one pre-trained model is provided.
12. A meaningful name has been used for the PR.


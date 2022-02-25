# OpenDR Toolkit Change Log

## Version 1.X
Released on XX, XXth, 2022.

  - New Features:
    - None.
  - Enhancements:
    - Added support for modular pip packages allowing tools to be installed separately ([#201](https://github.com/opendr-eu/opendr/pull/201)).
    - Simplified the installation process for pip by including the appropriate post-installation scripts ([#201](https://github.com/opendr-eu/opendr/pull/201)).
    - Improved the structure of the toolkit by moving `io` from `utils` to `engine.helper` ([#201](https://github.com/opendr-eu/opendr/pull/201)).
    - Added support for `post-install` scripts and `opendr` dependencies in `.ini` files  ([#201](https://github.com/opendr-eu/opendr/pull/201)). 
    - Updated toolkit to support CUDA 11.1
  - Bug Fixes:
    - Updated wheel building pipeline to include missing files and removed unnecessary dependencies ([#200](https://github.com/opendr-eu/opendr/pull/200)).
  - Dependency Updates:
    - `heart anomaly detection`: upgraded scikit-learn runtime dependency from 0.21.3 to 0.22 ([#198](https://github.com/opendr-eu/opendr/pull/198)).
    - Relaxed all dependencies to allow future versions of non-critical tools to be used ([#201](https://github.com/opendr-eu/opendr/pull/201)).


## Version 1.0
Released on December 31th, 2021.

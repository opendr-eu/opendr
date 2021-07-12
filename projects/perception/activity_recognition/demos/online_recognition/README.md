# OpenDR human activity recognition demo
<div align="left">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" height="20">
  </a>
</div>

Live demo of online human activity recognition using the [OpenDR toolkit](https://opendr.eu).
It captures the video stream from a webcam, performs frame-by-frame predictions, and presents the results on a web UI.


## Set-up
After setting up the _OpenDR toolkit_, install dependencies of this demo by navigating to this folder and run:
```bash
pip install -e .
```


## Running the example
Human Activity Recognition using [X3D](https://openaccess.thecvf.com/content_CVPR_2020/papers/Feichtenhofer_X3D_Expanding_Architectures_for_Efficient_Video_Recognition_CVPR_2020_paper.pdf)
```bash
python demo.py --ip 0.0.0.0 --port 8000 --algorithm x3d --model xs
```

Human Activity Recognition using CoX3D
```bash
python demo.py --ip 0.0.0.0 --port 8000 --algorithm cox3d --model s
```

If you navigate your piano and http://0.0.0.0:8000 and pick up a ukulele, you might see something like this:

<img src="activity_recognition/video.gif">

For other options, see `python demo.py --help`


## Troubleshooting
If no video is displayed, you may try to select another video source using the `--video_source` flag:
```bash
python demo.py --ip 0.0.0.0 --port 8000 --algorithm cox3d --model s --video_source 1
```

## Acknowledgement
This work has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 871449 (OpenDR). This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.

# OpenDR C API Nanodet Demo

C API implementation of nanodet models for inference.
To run the demo, the downloaded model can be used or it can be exported with JIT optimization from the python implementation, see [Nanodet optimization](../../../../../docs/reference/object-detection-2d-nanodet.md#nanodetlearneroptimize).

After installation, the demo can be run from projects/c_api directory with:
```sh
./build/nanodet_libtorch_demo ./path/to/your/model.pth device_name{cpu, cuda} ./path/to/your/image.jpg height width
```

Or with the downloaded model and image with:

```sh
./build/nanodet_libtorch_demo ./data/object_detection_2d/nanodet/optimized_model/nanodet_m.pth cuda ./data/object_detection_2d/nanodet/database/000000000036.jpg 320 320
```

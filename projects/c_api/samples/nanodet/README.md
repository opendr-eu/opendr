# OpenDR C API Nanodet Demo

C API implementation of nanodet models for inference. To use the models first must be exported with the optimization Jit from python.
After the installation can be run from projects/c_api directory with:
```sh
./built/nanodet_libtorch_demo ./path/to/your/model.pth device_name{cpu, cuda} ./path/to/your/image.jpg height width
```

After installation a temporal model and image are downloaded based on nanodet_m model from python.
You can run it as:

```sh
./built/nanodet_libtorch_demo ./data/nanodet/optimized_model/nanodet_m.pth cuda ./data/nanodet/database/000000000036.jpg 320 320
```

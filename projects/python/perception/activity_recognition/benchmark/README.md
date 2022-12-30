# Human Activity Recognition Benchmark
This folder contains a script for benchmarking the inference of Human Activity Recognition learners found at
```python
from opendr.perception.activity_recognition import X3DLearner
from opendr.perception.activity_recognition import CoX3DLearner
```

The script include logging of FLOPS and params for `learner.model`, inference timing, and energy-consumption (NVIDIA Jetson only).

The benchmarking runs twice; Once using `learner.infer` and once using `learner.model.forward`. The results of each are printed accordingly.


## Setup
Please install [`pytorch-benchmark`](https://github.com/LukasHedegaard/pytorch-benchmark):
```bash
pip install pytorch-benchmark
```

## Running the benchmark
X3D
```bash
./benchmark_x3d.py
```

CoX3D
```bash
./benchmark_cox3d.py
```

CoTransEnc
```bash
./benchmark_cotransenc.py
```
NB: The CoTransEnc module benchmarks various configurations of the Continual Transformer Encoder modules only. This doesn't include any feature-extraction that you might want to use beforehand.
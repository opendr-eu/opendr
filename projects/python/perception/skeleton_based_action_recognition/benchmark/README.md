# Skeleton-based Human Activity Recognition Benchmark
This folder contains a script for benchmarking the inference of Skeleton-based Human Activity Recognition learners.

The script include logging of FLOPS and params for `learner.model`, inference timing, and energy-consumption (NVIDIA Jetson only).

The benchmarking runs twice; Once using `learner.infer` and once using `learner.model.forward`. The results of each are printed accordingly.


### Setup
Please install [`pytorch-benchmark`](https://github.com/LukasHedegaard/pytorch-benchmark):
```bash
pip install pytorch-benchmark
```

### Running the benchmark
Benchmark the SpatioTemporalGCNLearner and ProgressiveSpatioTemporalGCNLearner
```bash
./benchmark_costgcn.py
```

Benchmarkk the CoSTGCNLearner with backbones: "costgcn", "coagcn", and "costr"
```bash
./benchmark_costgcn.py
```
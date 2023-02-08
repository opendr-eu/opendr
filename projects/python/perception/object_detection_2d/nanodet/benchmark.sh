#!/bin/sh
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model EfficientNet_Lite0_320 > ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model EfficientNet_Lite1_416 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model EfficientNet_Lite2_512 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model RepVGG_A0_416 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model g >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model t >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model m >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model m_0.5x >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model m_1.5x >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model m_416 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model m_1.5x_416 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model plus_m_320 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model plus_m_1.5x_320 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model plus_m_416 >> ./measures.txt
python3 ./benchmark_demo.py --optimize-onnx --optimize-jit --model plus_m_1.5x_416 >> ./measures.txt

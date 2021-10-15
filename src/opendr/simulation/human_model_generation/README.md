# Human Model Generation from Image/s

This folder contains code for:
*  3D human model generation from images.
*  3D pose approximation of human models.
*  Synthetic multi-view image generation for human-centric perception tasks.

## Sources

Code for [PIFu](https://arxiv.org/abs/1905.05172) is taken from [https://github.com/shunsukesaito/PIFu](https://github.com/shunsukesaito/PIFu).
Some major changes are referenced below:
*  The original code is modified in order to be compatible with the OpenDR specifications.
*  The ```PIFuGeneratorLearner``` supports only inference with pretrained models.
*  Some extra utilities have been added including the 3D pose approximation of human models and the generation of multi-view renderings of the human models.

Part of the original code is located in ```utilities/PIFu``` directory and is licensed under the MIT license:
```
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

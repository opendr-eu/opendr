#include <tuple>

#include <torch/extension.h>
#include <torch/torch.h>

#include "utils/checks.h"
#include "roi_sampling.h"

std::tuple<at::Tensor, at::Tensor> roi_sampling_forward(
    const at::Tensor& x, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int> out_size,
    Interpolation interpolation, PaddingMode padding, bool valid_mask) {
  // Check dimensions
  TORCH_CHECK(x.ndimension() == 4, "x must be a 4-dimensional tensor");
  TORCH_CHECK(bbx.ndimension() == 2, "bbx must be a 2-dimensional tensor");
  TORCH_CHECK(idx.ndimension() == 1, "idx must be a 1-dimensional tensor");
  TORCH_CHECK(bbx.size(0) == idx.size(0), "idx and bbx must have the same size in the first dimension");
  TORCH_CHECK(bbx.size(1) == 4, "bbx must be N x 4");

  // Check types
  TORCH_CHECK(bbx.scalar_type() == at::ScalarType::Float, "bbx must have type float32");
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Long, "idx must have type long");

  if (x.is_cuda()) {
    CHECK_CUDA(bbx);
    CHECK_CUDA(idx);

    return roi_sampling_forward_cuda(x, bbx, idx, out_size, interpolation, padding, valid_mask);
  } else {
    CHECK_CPU(bbx);
    CHECK_CPU(idx);

    return roi_sampling_forward_cpu(x, bbx, idx, out_size, interpolation, padding, valid_mask);
  }
}

at::Tensor roi_sampling_backward(
    const at::Tensor& dy, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int, int> in_size,
    Interpolation interpolation, PaddingMode padding) {
  // Check dimensions
  TORCH_CHECK(dy.ndimension() == 4, "dy must be a 4-dimensional tensor");
  TORCH_CHECK(bbx.ndimension() == 2, "bbx must be a 2-dimensional tensor");
  TORCH_CHECK(idx.ndimension() == 1, "idx must be a 1-dimensional tensor");
  TORCH_CHECK(bbx.size(0) == idx.size(0), "idx and bbx must have the same size in the first dimension");
  TORCH_CHECK(bbx.size(1) == 4, "bbx must be N x 4");

  // Check types
  TORCH_CHECK(bbx.scalar_type() == at::ScalarType::Float, "bbx must have type float32");
  TORCH_CHECK(idx.scalar_type() == at::ScalarType::Long, "idx must have type long");

  if (dy.is_cuda()) {
    CHECK_CUDA(bbx);
    CHECK_CUDA(idx);

    return roi_sampling_backward_cuda(dy, bbx, idx, in_size, interpolation, padding);
  } else {
    CHECK_CPU(bbx);
    CHECK_CPU(idx);

    return roi_sampling_backward_cpu(dy, bbx, idx, in_size, interpolation, padding);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::enum_<PaddingMode>(m, "PaddingMode")
    .value("Zero", PaddingMode::Zero)
    .value("Border", PaddingMode::Border);

  pybind11::enum_<Interpolation>(m, "Interpolation")
    .value("Bilinear", Interpolation::Bilinear)
    .value("Nearest", Interpolation::Nearest);

  m.def("roi_sampling_forward", &roi_sampling_forward, "ROI sampling forward");
  m.def("roi_sampling_backward", &roi_sampling_backward, "ROI sampling backward");
}

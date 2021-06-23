#include <vector>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/detail/IndexUtils.cuh>

#include "utils/checks.h"
#include "utils/cuda.cuh"
#include "utils/common.h"
#include "roi_sampling.h"


template<typename scalar_t, typename coord_t, typename index_t, typename Sampler>
__global__ void roi_sampling_forward_kernel(
    const at::PackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> x,
    const at::PackedTensorAccessor<coord_t, 2, at::RestrictPtrTraits, index_t> bbx,
    const at::PackedTensorAccessor<int64_t, 1, at::RestrictPtrTraits, index_t> idx,
    at::PackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> y,
    at::PackedTensorAccessor<uint8_t, 3, at::RestrictPtrTraits, index_t> mask,
    bool valid_mask,
    Sampler sampler) {

  // Dimensions
  auto chn = x.size(1), img_height = x.size(2), img_width = x.size(3);
  auto roi_height = y.size(2), roi_width = y.size(3);
  index_t sizes[3] = {chn, roi_height, roi_width};
  index_t out_size = chn * roi_height * roi_width;

  index_t n = blockIdx.x;

  // Get bounding box coordinates and image index
  auto i0 = bbx[n][0], j0 = bbx[n][1], i1 = bbx[n][2], j1 = bbx[n][3];
  auto img_idx = idx[n];

  auto x_n = x[img_idx], y_n = y[n];

  for (int iter = threadIdx.x; iter < out_size; iter += blockDim.x) {
    // Find current indices
    index_t c, i, j;
    ind2sub<index_t, 3>(iter, sizes, j, i, c);

    auto y_img = roi_to_img(static_cast<coord_t>(i) + coord_t(0.5), i0, i1, static_cast<coord_t>(roi_height));
    auto x_img = roi_to_img(static_cast<coord_t>(j) + coord_t(0.5), j0, j1, static_cast<coord_t>(roi_width));

    y_n[c][i][j] = sampler.forward(y_img, x_img, x_n[c]);

    if (valid_mask) {
      mask[n][i][j] =
          y_img >= 0 && y_img < static_cast<coord_t>(img_height) &&
          x_img >= 0 && x_img < static_cast<coord_t>(img_width);
    }
  }
}

template<typename scalar_t, typename coord_t, typename index_t>
void roi_sampling_forward_template(
    const at::Tensor& x, const at::Tensor& bbx, const at::Tensor& idx, at::Tensor& y, at::Tensor& mask,
    Interpolation interpolation, PaddingMode padding, bool valid_mask) {
  // Create accessors
  auto x_accessor = x.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>();
  auto bbx_accessor = bbx.packed_accessor<coord_t, 2, at::RestrictPtrTraits, index_t>();
  auto idx_accessor = idx.packed_accessor<int64_t, 1, at::RestrictPtrTraits, index_t>();
  auto y_accessor = y.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>();
  auto mask_accessor = mask.packed_accessor<uint8_t, 3, at::RestrictPtrTraits, index_t>();

  dim3 blocks(y.size(0));
  dim3 threads(getNumThreads(y.size(1) * y.size(2) * y.size(3)));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Run kernel
  DISPATCH_INTERPOLATION_PADDING_MODES(interpolation, padding, ([&] {
    indexer_t indexer(x.size(2), x.size(3));
    interpolator_t interpolator;
    sampler_t sampler(indexer, interpolator);

    roi_sampling_forward_kernel<scalar_t, coord_t, index_t, sampler_t><<<blocks, threads, 0, stream>>>(
        x_accessor, bbx_accessor, idx_accessor, y_accessor, mask_accessor, valid_mask, sampler);
  }));
}

std::tuple<at::Tensor, at::Tensor> roi_sampling_forward_cuda(
    const at::Tensor& x, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int> out_size,
    Interpolation interpolation, PaddingMode padding, bool valid_mask) {

  // Prepare outputs
  auto y = at::empty({idx.size(0), x.size(1), std::get<0>(out_size), std::get<1>(out_size)}, x.options());
  auto mask = valid_mask
      ? at::zeros({idx.size(0), std::get<0>(out_size), std::get<1>(out_size)}, x.options().dtype(at::kByte))
      : at::zeros({1, 1, 1}, x.options().dtype(at::kByte));

  AT_DISPATCH_ALL_TYPES(x.scalar_type(), "roi_sampling_forward_cuda", ([&] {
    if (at::cuda::detail::canUse32BitIndexMath(x) && at::cuda::detail::canUse32BitIndexMath(y)) {
      roi_sampling_forward_template<scalar_t, float, int32_t>(
          x, bbx, idx, y, mask, interpolation, padding, valid_mask);
    } else {
      roi_sampling_forward_template<scalar_t, float, int64_t>(
          x, bbx, idx, y, mask, interpolation, padding, valid_mask);
    }
  }));

  return std::make_tuple(y, mask);
}

template<typename scalar_t, typename coord_t, typename index_t, typename Sampler>
__global__ void roi_sampling_backward_kernel(
    const at::PackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> dy,
    const at::PackedTensorAccessor<coord_t, 2, at::RestrictPtrTraits, index_t> bbx,
    const at::PackedTensorAccessor<int64_t, 1, at::RestrictPtrTraits, index_t> idx,
    at::PackedTensorAccessor<scalar_t, 4, at::RestrictPtrTraits, index_t> dx,
    Sampler sampler) {

  // Dimensions
  auto num = dy.size(0), roi_height = dy.size(2), roi_width = dy.size(3);
  auto img_height = dx.size(2), img_width = dx.size(3);
  index_t iter_sizes[3] = {num, roi_height, roi_width};
  index_t iter_size = num * roi_height * roi_width;

  // Local indices
  index_t c = blockIdx.x;

  for (int iter = threadIdx.x; iter < iter_size; iter += blockDim.x) {
    // Find current indices
    index_t n, i, j;
    ind2sub<index_t, 3>(iter, iter_sizes, j, i, n);

    // Get bounding box coordinates and image index
    // Get bounding box coordinates and image index
    auto i0 = bbx[n][0], j0 = bbx[n][1], i1 = bbx[n][2], j1 = bbx[n][3];
    auto img_idx = idx[n];

    auto y_img = roi_to_img(static_cast<coord_t>(i) + coord_t(0.5), i0, i1, static_cast<coord_t>(roi_height));
    auto x_img = roi_to_img(static_cast<coord_t>(j) + coord_t(0.5), j0, j1, static_cast<coord_t>(roi_width));

    sampler.backward(y_img, x_img, dy[n][c][i][j], dx[img_idx][c]);
  }
}

template<typename scalar_t, typename coord_t, typename index_t>
void roi_sampling_backward_template(
    const at::Tensor& dy, const at::Tensor& bbx, const at::Tensor& idx, at::Tensor& dx,
    Interpolation interpolation, PaddingMode padding) {
  // Create accessors
  auto dy_accessor = dy.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>();
  auto bbx_accessor = bbx.packed_accessor<coord_t, 2, at::RestrictPtrTraits, index_t>();
  auto idx_accessor = idx.packed_accessor<int64_t, 1, at::RestrictPtrTraits, index_t>();
  auto dx_accessor = dx.packed_accessor<scalar_t, 4, at::RestrictPtrTraits, index_t>();

  dim3 blocks(dy.size(1));
  dim3 threads(getNumThreads(dy.size(0) * dy.size(2) * dy.size(3)));
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  // Run kernel
  DISPATCH_INTERPOLATION_PADDING_MODES(interpolation, padding, ([&] {
    indexer_t indexer(dx.size(2), dx.size(3));
    interpolator_t interpolator;
    sampler_t sampler(indexer, interpolator);

    roi_sampling_backward_kernel<scalar_t, coord_t, index_t, sampler_t><<<blocks, threads, 0, stream>>>(
        dy_accessor, bbx_accessor, idx_accessor, dx_accessor, sampler);
  }));
}

at::Tensor roi_sampling_backward_cuda(
    const at::Tensor& dy, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int, int> in_size,
    Interpolation interpolation, PaddingMode padding) {

  // Prepare output
  auto dx = at::zeros({std::get<0>(in_size), dy.size(1), std::get<1>(in_size), std::get<2>(in_size)}, dy.options());

  AT_DISPATCH_FLOATING_TYPES(dy.scalar_type(), "roi_sampling_backward_cuda", ([&] {
    if (at::cuda::detail::canUse32BitIndexMath(dy) && at::cuda::detail::canUse32BitIndexMath(dx)) {
      roi_sampling_backward_template<scalar_t, float, int32_t>(
          dy, bbx, idx, dx, interpolation, padding);
    } else {
      roi_sampling_backward_template<scalar_t, float, int64_t>(
          dy, bbx, idx, dx, interpolation, padding);
    }
  }));

  return dx;
}
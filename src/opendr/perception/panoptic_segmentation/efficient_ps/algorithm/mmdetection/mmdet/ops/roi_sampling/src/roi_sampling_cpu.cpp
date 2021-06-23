#include <ATen/ATen.h>

#include "roi_sampling.h"

template<typename scalar_t, typename coord_t, typename Sampler>
void roi_sampling_forward_impl(
    at::TensorAccessor<scalar_t, 4> x,
    at::TensorAccessor<coord_t, 2> bbx,
    at::TensorAccessor<int64_t, 1> idx,
    at::TensorAccessor<scalar_t, 4> y,
    at::TensorAccessor<uint8_t, 3> mask,
    bool valid_mask,
    Sampler sampler) {
  auto roi_height = static_cast<coord_t>(y.size(2)),
       roi_width  = static_cast<coord_t>(y.size(3));
  auto img_height = static_cast<coord_t>(x.size(2)),
       img_width  = static_cast<coord_t>(x.size(3));

  for (int64_t n = 0; n < idx.size(0); ++n) {
    auto img_idx = idx[n];
    auto i0 = bbx[n][0], j0 = bbx[n][1], i1 = bbx[n][2], j1 = bbx[n][3];

    for (int64_t c = 0; c < x.size(1); ++c) {
      // Create indexer for this plane and image
      auto accessor = x[img_idx][c];

      for (int64_t i_roi = 0; i_roi < y.size(2); ++i_roi) {
        auto y_img = roi_to_img(static_cast<coord_t>(i_roi) + coord_t(0.5), i0, i1, roi_height);

        for (int64_t j_roi = 0; j_roi < y.size(3); ++j_roi) {
          auto x_img = roi_to_img(static_cast<coord_t>(j_roi) + coord_t(0.5), j0, j1, roi_width);

          y[n][c][i_roi][j_roi] = sampler.forward(y_img, x_img, accessor);

          // Optionally write to mask
          if (valid_mask) {
            mask[n][i_roi][j_roi] = y_img >= 0 && y_img < img_height && x_img >= 0 && x_img < img_width;
          }
        }
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> roi_sampling_forward_cpu(
    const at::Tensor& x, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int> out_size,
    Interpolation interpolation, PaddingMode padding, bool valid_mask) {

  // Prepare outputs
  auto y = at::empty({idx.size(0), x.size(1), std::get<0>(out_size), std::get<1>(out_size)}, x.options());
  auto mask = valid_mask
      ? at::zeros({idx.size(0), std::get<0>(out_size), std::get<1>(out_size)}, x.options().dtype(at::kByte))
      : at::zeros({1, 1, 1}, x.options().dtype(at::kByte));

  AT_DISPATCH_ALL_TYPES(x.scalar_type(), "roi_sampling_forward_cpu", ([&] {
    using coord_t = float;
    using index_t = int64_t;

    auto _x = x.accessor<scalar_t, 4>();
    auto _bbx = bbx.accessor<coord_t, 2>();
    auto _idx = idx.accessor<index_t, 1>();
    auto _y = y.accessor<scalar_t, 4>();
    auto _mask = mask.accessor<uint8_t, 3>();

    DISPATCH_INTERPOLATION_PADDING_MODES(interpolation, padding, ([&] {
      indexer_t indexer(x.size(2), x.size(3));
      interpolator_t interpolator;
      sampler_t sampler(indexer, interpolator);

      roi_sampling_forward_impl<scalar_t, coord_t, sampler_t>(_x, _bbx, _idx, _y, _mask, valid_mask, sampler);
    }));
  }));

  return std::make_tuple(y, mask);
}

template<typename scalar_t, typename coord_t, typename Sampler>
void roi_sampling_backward_impl(
    at::TensorAccessor<scalar_t, 4> dy,
    at::TensorAccessor<coord_t, 2> bbx,
    at::TensorAccessor<int64_t, 1> idx,
    at::TensorAccessor<scalar_t, 4> dx,
    Sampler sampler) {
  auto roi_height = static_cast<coord_t>(dy.size(2)),
       roi_width  = static_cast<coord_t>(dy.size(3));

  for (int64_t n = 0; n < idx.size(0); ++n) {
    auto img_idx = idx[n];
    auto i0 = bbx[n][0], j0 = bbx[n][1], i1 = bbx[n][2], j1 = bbx[n][3];

    for (int64_t c = 0; c < dy.size(1); ++c) {
      // Create indexer for this plane and image
      auto accessor = dx[img_idx][c];

      for (int64_t i_roi = 0; i_roi < dy.size(2); ++i_roi) {
        auto y_img = roi_to_img(static_cast<coord_t>(i_roi) + coord_t(0.5), i0, i1, roi_height);

        for (int64_t j_roi = 0; j_roi < dy.size(3); ++j_roi) {
          auto x_img = roi_to_img(static_cast<coord_t>(j_roi) + coord_t(0.5), j0, j1, roi_width);

          sampler.backward(y_img, x_img, dy[n][c][i_roi][j_roi], accessor);
        }
      }
    }
  }
}

at::Tensor roi_sampling_backward_cpu(
    const at::Tensor& dy, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int, int> in_size,
    Interpolation interpolation, PaddingMode padding) {

  // Prepare output
  auto dx = at::zeros({std::get<0>(in_size), dy.size(1), std::get<1>(in_size), std::get<2>(in_size)}, dy.options());

  AT_DISPATCH_ALL_TYPES(dy.scalar_type(), "roi_sampling_backward_cpu", ([&] {
    using coord_t = float;
    using index_t = int64_t;

    auto _dy = dy.accessor<scalar_t, 4>();
    auto _bbx = bbx.accessor<coord_t, 2>();
    auto _idx = idx.accessor<index_t, 1>();
    auto _dx = dx.accessor<scalar_t, 4>();

    DISPATCH_INTERPOLATION_PADDING_MODES(interpolation, padding, ([&] {
      indexer_t indexer(dx.size(2), dx.size(3));
      interpolator_t interpolator;
      sampler_t sampler(indexer, interpolator);

      roi_sampling_backward_impl<scalar_t, coord_t, sampler_t>(_dy, _bbx, _idx, _dx, sampler);
    }));
  }));

  return dx;
}

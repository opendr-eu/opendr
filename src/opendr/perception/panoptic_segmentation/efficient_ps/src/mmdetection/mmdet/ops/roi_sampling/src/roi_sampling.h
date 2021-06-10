#pragma once

#include <type_traits>
#include <tuple>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include "utils/common.h"

// ENUMS

enum class PaddingMode { Zero, Border };
enum class Interpolation { Bilinear, Nearest };

// PROTOTYPES

std::tuple<at::Tensor, at::Tensor> roi_sampling_forward_cpu(
    const at::Tensor& x, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int> out_size,
    Interpolation interpolation, PaddingMode padding, bool valid_mask);
std::tuple<at::Tensor, at::Tensor> roi_sampling_forward_cuda(
    const at::Tensor& x, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int> out_size,
    Interpolation interpolation, PaddingMode padding, bool valid_mask);

at::Tensor roi_sampling_backward_cpu(
    const at::Tensor& dy, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int, int> in_size,
    Interpolation interpolation, PaddingMode padding);
at::Tensor roi_sampling_backward_cuda(
    const at::Tensor& dy, const at::Tensor& bbx, const at::Tensor& idx, std::tuple<int, int, int> in_size,
    Interpolation interpolation, PaddingMode padding);

/* CONVENTIONS
 *
 * Integer indexes are i (vertical), j (horizontal) and k (generic)
 * Continuous coordinates are y (vertical), x (horizontal) and s (generic)
 *
 * The relation between the two is: y = i + 0.5, x = j + 0.5
 */

// SAMPLER

template<typename scalar_t, typename coord_t, typename index_t, typename Indexer, typename Interpolator>
struct Sampler {
  Sampler(Indexer indexer, Interpolator interpolator) : _indexer(indexer), _interpolator(interpolator) {}

  template<typename Accessor>
  HOST_DEVICE scalar_t forward(coord_t y, coord_t x, Accessor accessor) const {
    // Step 1: find the four indices of the points to read from the input and their offsets
    index_t i_l, i_h, j_l, j_h;
    coord_t delta_y, delta_x;
    _neighbors(y, i_l, i_h, delta_y);
    _neighbors(x, j_l, j_h, delta_x);

    // Step 2: read the four points
    scalar_t p_ll = _indexer.get(accessor, i_l, j_l),
             p_lh = _indexer.get(accessor, i_l, j_h),
             p_hl = _indexer.get(accessor, i_h, j_l),
             p_hh = _indexer.get(accessor, i_h, j_h);

    // Step 3: get the interpolated value
    return _interpolator.get(delta_y, delta_x, p_ll, p_lh, p_hl, p_hh);
  }

  template<typename Accessor>
  HOST_DEVICE void backward(coord_t y, coord_t x, scalar_t grad, Accessor accessor) const {
    // Step 1: find the four indices of the points to read from the input and their offsets
    index_t i_l, i_h, j_l, j_h;
    coord_t delta_y, delta_x;
    _neighbors(y, i_l, i_h, delta_y);
    _neighbors(x, j_l, j_h, delta_x);

    // Step 2: reverse-interpolation
    scalar_t p_ll, p_lh, p_hl, p_hh;
    _interpolator.set(delta_y, delta_x, grad, p_ll, p_lh, p_hl, p_hh);

    // Step 3: accumulate
    _indexer.set(accessor, i_l, j_l, p_ll);
    _indexer.set(accessor, i_l, j_h, p_lh);
    _indexer.set(accessor, i_h, j_l, p_hl);
    _indexer.set(accessor, i_h, j_h, p_hh);
  }

 private:
  INLINE_HOST_DEVICE void _neighbors(coord_t s, index_t &k_l, index_t &k_h, coord_t &delta) const {
    k_l = static_cast<index_t>(FLOOR(s - 0.5));
    k_h = k_l + 1;
    delta = s - (static_cast<coord_t>(k_l) + 0.5);
  }

 private:
  Indexer _indexer;
  Interpolator _interpolator;
};

// INDEXER

template<typename index_t>
struct IndexerBase {
  IndexerBase(index_t height, index_t width) : _height(height), _width(width) {};

  index_t _height;
  index_t _width;
};

template<typename scalar_t, typename index_t, PaddingMode padding>
struct Indexer;

template<typename scalar_t, typename index_t>
struct Indexer<scalar_t, index_t, PaddingMode::Zero> : IndexerBase<index_t> {
  using IndexerBase<index_t>::IndexerBase;

  template<typename Accessor>
  INLINE_HOST_DEVICE scalar_t get(Accessor accessor, index_t i, index_t j) const {
    return _in_bounds(i, this->_height) && _in_bounds(j, this->_width) ? accessor[i][j] : 0;
  }

  template<typename Accessor>
  INLINE_HOST_DEVICE void set(Accessor accessor, index_t i, index_t j, scalar_t value) const {
    if (_in_bounds(i, this->_height) && _in_bounds(j, this->_width)) {
      ACCUM_BLOCK(accessor[i][j], value);
    }
  }

 private:
  INLINE_HOST_DEVICE bool _in_bounds(index_t k, index_t size) const {
    return k >= 0 && k < size;
  }
};

template<typename scalar_t, typename index_t>
struct Indexer<scalar_t, index_t, PaddingMode::Border> : IndexerBase<index_t> {
  using IndexerBase<index_t>::IndexerBase;

  template<typename Accessor>
  INLINE_HOST_DEVICE scalar_t get(Accessor accessor, index_t i, index_t j) const {
    _clamp(i, j);
    return accessor[i][j];
  }

  template<typename Accessor>
  INLINE_HOST_DEVICE void set(Accessor accessor, index_t i, index_t j, scalar_t value) const {
    _clamp(i, j);
    ACCUM_BLOCK(accessor[i][j], value);
  }

 private:
  INLINE_HOST_DEVICE void _clamp(index_t &i, index_t &j) const {
    i = i >= 0      ? i : 0;
    i = i < this->_height ? i : this->_height - 1;
    j = j >= 0      ? j : 0;
    j = j < this->_width  ? j : this->_width - 1;
  }
};

// INTERPOLATORS

template<typename scalar_t, typename coord_t, Interpolation interpolation>
struct Interpolator;

template<typename scalar_t, typename coord_t>
struct Interpolator<scalar_t, coord_t, Interpolation::Bilinear> {
  INLINE_HOST_DEVICE scalar_t get(
      coord_t delta_y, coord_t delta_x, scalar_t p_ll, scalar_t p_lh, scalar_t p_hl, scalar_t p_hh) const {
    scalar_t hor_int_l = (1 - delta_x) * p_ll + delta_x * p_lh;
    scalar_t hor_int_h = (1 - delta_x) * p_hl + delta_x * p_hh;
    return (1 - delta_y) * hor_int_l + delta_y * hor_int_h;
  }

  INLINE_HOST_DEVICE void set(
      coord_t delta_y, coord_t delta_x, scalar_t value,
      scalar_t &p_ll, scalar_t &p_lh, scalar_t &p_hl, scalar_t &p_hh) const {
    p_ll = (1 - delta_x) * (1 - delta_y) * value;
    p_lh = delta_x       * (1 - delta_y) * value;
    p_hl = (1 - delta_x) *       delta_y * value;
    p_hh = delta_x       *       delta_y * value;
  }
};

template<typename scalar_t, typename coord_t>
struct Interpolator<scalar_t, coord_t, Interpolation::Nearest> {
  INLINE_HOST_DEVICE scalar_t get(
      coord_t delta_y, coord_t delta_x, scalar_t p_ll, scalar_t p_lh, scalar_t p_hl, scalar_t p_hh) const {
    return p_ll * static_cast<scalar_t>(delta_y <  0.5 && delta_x <  0.5) +
           p_lh * static_cast<scalar_t>(delta_y <  0.5 && delta_x >= 0.5) +
           p_hl * static_cast<scalar_t>(delta_y >= 0.5 && delta_x <  0.5) +
           p_hh * static_cast<scalar_t>(delta_y >= 0.5 && delta_x >= 0.5);
  }

  INLINE_HOST_DEVICE void set(
      coord_t delta_y, coord_t delta_x, scalar_t value,
      scalar_t &p_ll, scalar_t &p_lh, scalar_t &p_hl, scalar_t &p_hh) const {
    p_ll = static_cast<scalar_t>(delta_y <  0.5 && delta_x <  0.5) * value;
    p_lh = static_cast<scalar_t>(delta_y <  0.5 && delta_x >= 0.5) * value;
    p_hl = static_cast<scalar_t>(delta_y >= 0.5 && delta_x <  0.5) * value;
    p_hh = static_cast<scalar_t>(delta_y >= 0.5 && delta_x >= 0.5) * value;
  }
};

// UTILITY FUNCTIONS AND MACROS

template<typename coord_t>
INLINE_HOST_DEVICE coord_t roi_to_img(coord_t s_roi, coord_t s0_img, coord_t s1_img, coord_t roi_size) {
  return s_roi / roi_size * (s1_img - s0_img) + s0_img;
}

template<typename coord_t>
INLINE_HOST_DEVICE coord_t img_to_img(coord_t s, coord_t size_in, coord_t size_out) {
  return s / size_in * size_out;
}

#define INTERPOLATION_PADDING_DEFINES(INTERPOLATION, PADDING)                       \
  using indexer_t = Indexer<scalar_t, index_t, PADDING>;                            \
  using interpolator_t = Interpolator<scalar_t, coord_t, INTERPOLATION>;            \
  using sampler_t = Sampler<scalar_t, coord_t, index_t, indexer_t, interpolator_t>;

#define DISPATCH_INTERPOLATION_PADDING_MODES(INTERPOLATION, PADDING, ...)        \
[&] {                                                                            \
  switch (INTERPOLATION) {                                                       \
  case Interpolation::Bilinear:                                                  \
    TORCH_CHECK(!std::is_integral<scalar_t>::value,                                 \
             "Bilinear interpolation is not available for integral types");      \
    switch (PADDING) {                                                           \
    case PaddingMode::Zero: {                                                    \
      INTERPOLATION_PADDING_DEFINES(Interpolation::Bilinear, PaddingMode::Zero)  \
      return __VA_ARGS__();                                                      \
    }                                                                            \
    case PaddingMode::Border: {                                                  \
      INTERPOLATION_PADDING_DEFINES(Interpolation::Bilinear, PaddingMode::Border)\
      return __VA_ARGS__();                                                      \
    }}                                                                           \
  case Interpolation::Nearest:                                                   \
    switch (PADDING) {                                                           \
    case PaddingMode::Zero: {                                                    \
      INTERPOLATION_PADDING_DEFINES(Interpolation::Nearest, PaddingMode::Zero)   \
      return __VA_ARGS__();                                                      \
    }                                                                            \
    case PaddingMode::Border: {                                                  \
      INTERPOLATION_PADDING_DEFINES(Interpolation::Nearest, PaddingMode::Border) \
      return __VA_ARGS__();                                                      \
    }}                                                                           \
  }                                                                              \
}()

#pragma once

#include <type_traits>
#include <ATen/ATen.h>

/*
 * Functions to share code between CPU and GPU
 */

#ifdef __CUDACC__
// CUDA versions

#define HOST_DEVICE __host__ __device__
#define INLINE_HOST_DEVICE __host__ __device__ inline
#define FLOOR(x) floor(x)

#if __CUDA_ARCH__ >= 600
// Recent compute capabilities have both grid-level and block-level atomicAdd for all data types, so we use those
#define ACCUM_BLOCK(x,y) atomicAdd_block(&(x),(y))
#define ACCUM(x, y) atomicAdd(&(x),(y))
#else
// Older architectures don't have block-level atomicAdd, nor atomicAdd for doubles, so we defer to atomicAdd for float
// and use the known atomicCAS-based implementation for double
template<typename data_t>
__device__ inline data_t atomic_add(data_t *address, data_t val) {
  return atomicAdd(address, val);
}

template<>
__device__ inline double atomic_add(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#define ACCUM_BLOCK(x,y) atomic_add(&(x),(y))
#define ACCUM(x,y) atomic_add(&(x),(y))
#endif // #if __CUDA_ARCH__ >= 600

#else
// CPU versions

#define HOST_DEVICE
#define INLINE_HOST_DEVICE inline
#define FLOOR(x) std::floor(x)
#define ACCUM_BLOCK(x,y) (x) += (y)
#define ACCUM(x,y) (x) += (y)

#endif // #ifdef __CUDACC__

/*
 * Other utility functions
 */
template<typename T, int dim>
INLINE_HOST_DEVICE void ind2sub(T i, T *sizes, T &i_n) {
  static_assert(dim == 1, "dim must be 1");
  i_n = i % sizes[0];
}

template<typename T, int dim, typename... Indices>
INLINE_HOST_DEVICE void ind2sub(T i, T *sizes, T &i_n, Indices&...args) {
  static_assert(dim == sizeof...(args) + 1, "dim must equal the number of args");
  i_n = i % sizes[dim - 1];
  ind2sub<T, dim - 1>(i / sizes[dim - 1], sizes, args...);
}

template<typename T> inline T div_up(T x, T y) {
  static_assert(std::is_integral<T>::value, "div_up is only defined for integral types");
  return x / y + (x % y > 0);
}
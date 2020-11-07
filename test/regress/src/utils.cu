#include "fwgpu/utils.hpp"

#include <cuda_runtime.h>

namespace fwgpu {

auto malloc_device(void **dptr, size_t size) -> int {
  auto retval = static_cast<int>(cudaMalloc(dptr, size));
  return retval;
}

auto malloc_unified(void **dptr, size_t size) -> int {
  auto retval = static_cast<int>(cudaMallocManaged(dptr, size));
  return retval;
}

auto memcpy_inferred(void *dest, const void *src, size_t size) -> int {
  auto retval = static_cast<int>(cudaMemcpy(dest, src, size, cudaMemcpyDefault));
  return retval;
}

auto free_device(void *dbuf) -> int {
  auto retval = static_cast<int>(cudaFree(dbuf));
  return retval;
}

auto memcpy_d2h(void *dest, const void *src, size_t size) -> int {
  auto retval = static_cast<int>(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
  return retval;
}

auto memcpy_h2d(void *dest, const void *src, size_t size) -> int {
  auto retval = static_cast<int>(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
  return retval;
}

auto memcpy_h2h(void *dest, const void *src, size_t size) -> int {
  auto retval = static_cast<int>(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice));
  return retval;
}

auto memcpy_d2d(void *dest, const void *src, size_t size) -> int {
  auto retval = static_cast<int>(cudaMemcpy(dest, src, size, cudaMemcpyHostToHost));
  return retval;
}

auto memcpy_2d_h2d(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int {
  auto retval = static_cast<int>(
      cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));
  return retval;
}

auto memcpy_2d_d2h(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int {
  auto retval = static_cast<int>(
      cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost));
  return retval;
}

auto memcpy_2d_d2d(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int {
  auto retval = static_cast<int>(
      cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice));
  return retval;
}

auto memcpy_2d_inferred(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int {
  auto retval = static_cast<int>(
      cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyDefault));
  return retval;
}

} // namespace fwgpu

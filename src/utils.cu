#include "fwgpu/utils.hpp"

namespace fwgpu {

auto malloc_device(void **dptr, size_t size) -> int {
  return static_cast<int>(cudaMalloc(dptr, size));
}

auto free_device(void *dbuf) -> int { return static_cast<int>(cudaFree(dbuf)); }

auto memcpy_d2h(void *dest, const void *src, size_t size) -> int {
  return static_cast<int>(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
}

auto memcpy_h2d(void *dest, const void *src, size_t size) -> int {
  return static_cast<int>(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
}

auto memcpy_2d_h2d(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int {
  return static_cast<int>(
      cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyHostToDevice));

}

auto memcpy_2d_d2h(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int {
  return static_cast<int>(
      cudaMemcpy2D(dest, dpitch, src, spitch, width, height, cudaMemcpyDeviceToHost));
}

} // namespace fwgpu

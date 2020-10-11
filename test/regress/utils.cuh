#ifndef cuASR_INTERNAL_UTILS
#define cuASR_INTERNAL_UTILS

#include <tuple>

#include "fwgpu/Matrix.hpp"

namespace fwgpu {
namespace internal {

template <typename T>
inline auto alloc_and_init_device_gemm_mats(
    const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C)
    -> std::tuple<T *, T *, T *> {
  // allocate for inputs and outputs on device
  void *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, A.bytesize());
  cudaMalloc(&d_B, B.bytesize());
  cudaMalloc(&d_C, C.bytesize());

  // copy inputs to device
  cudaMemcpy(d_A, A.get_buf(), A.bytesize(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.get_buf(), B.bytesize(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C.get_buf(), C.bytesize(), cudaMemcpyHostToDevice);

  return std::make_tuple(
      reinterpret_cast<T *>(d_A), reinterpret_cast<T *>(d_B), reinterpret_cast<T *>(d_C));
}

template <typename T>
inline auto dealloc_device_gemm_mats(std::tuple<T *, T *, T *> device_ptrs) -> void {
  cudaFree(std::get<0>(device_ptrs));
  cudaFree(std::get<1>(device_ptrs));
  cudaFree(std::get<2>(device_ptrs));
}

template <typename T>
inline auto alloc_and_init_device_gemm_mats(
    const Matrix<T> &A, const Matrix<T> &B, const Matrix<T> &C, const Matrix<T> &D)
    -> std::tuple<T *, T *, T *, T *> {
  // allocate for inputs and outputs on device
  void *d_A, *d_B, *d_C, *d_D;
  cudaMalloc(&d_A, A.bytesize());
  cudaMalloc(&d_B, B.bytesize());
  cudaMalloc(&d_C, C.bytesize());
  cudaMalloc(&d_D, D.bytesize());

  // copy inputs to device
  cudaMemcpy(d_A, A.get_buf(), A.bytesize(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.get_buf(), B.bytesize(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C.get_buf(), C.bytesize(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_D, D.get_buf(), D.bytesize(), cudaMemcpyHostToDevice);

  return std::make_tuple(
      reinterpret_cast<T *>(d_A), reinterpret_cast<T *>(d_B), reinterpret_cast<T *>(d_C),
      reinterpret_cast<T *>(d_D));
}

template <typename T>
inline auto dealloc_device_gemm_mats(std::tuple<T *, T *, T *, T *> device_ptrs) -> void {
  cudaFree(std::get<0>(device_ptrs));
  cudaFree(std::get<1>(device_ptrs));
  cudaFree(std::get<2>(device_ptrs));
  cudaFree(std::get<3>(device_ptrs));
}

} // namespace internal
} // namespace fwgpu

#endif // cuASR_INTERNAL_UTILS

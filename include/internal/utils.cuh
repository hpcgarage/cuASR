#ifndef FWGPU_INTERNAL_UTILS
#define FWGPU_INTERNAL_UTILS

namespace fwgpu {
namespace internal {
  template <typename T>
  inline auto alloc_gemm_mats_on_gpu(const Matrix<T> &A, const Matrix<T> &B)
      -> std::tuple<T *, T *, T *> {
    const auto m         = A.num_rows();
    const auto k         = A.num_cols(); // B.num_rows();
    const auto n         = B.num_cols();
    auto output_bytesize = m * n * sizeof(T);
    T *c_bytes           = new T[m * n];

    // allocate for inputs and outputs on device
    T *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.bytesize());
    cudaMalloc(&d_B, B.bytesize());
    cudaMalloc(&d_C, output_bytesize);

    // copy inputs to device
    cudaMemcpy(d_A, A.get_buf(), A.bytesize(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.get_buf(), B.bytesize(), cudaMemcpyHostToDevice);

    return std::make_tuple(d_A, d_B, d_C);
  }

  template <typename T>
  inline auto dealloc_gemm_mats_on_gpu(std::tuple<T *, T *, T *> device_ptrs) -> void {
    cudaFree(std::get<0>(device_ptrs));
    cudaFree(std::get<1>(device_ptrs));
    cudaFree(std::get<2>(device_ptrs));
  }
} // namespace internal
} // namespace fwgpu

#endif // FWGPU_INTERNAL_UTILS

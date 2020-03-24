#ifndef GPU_GEMM_BM_ENTRY
#define GPU_GEMM_BM_ENTRY

#include "fwgpu/Matrix.hpp"
#include "fwgpu/cpu_gemm.hpp"
#include "fwgpu/cpu_srgemm.hpp"

namespace fwgpu {
namespace testing {
  auto cublas_sgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float>;

  auto cutlass_sgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float>;

  auto cutlass_srsgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float>;

  auto gpu_sgemm_naive_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float>;

  auto gpu_srgemm_naive_entry(const Matrix<float> &A, const Matrix<float> &B)
      -> Matrix<float>;

  template <typename T>
  inline auto cpu_gemm_naive_entry(const Matrix<T> &A, const Matrix<T> &B) -> Matrix<T> {
    const auto m = A.num_rows();
    const auto k = A.num_cols(); // B.num_rows();
    const auto n = B.num_cols();
    auto C       = Matrix<T>(m, n);
    fwgpu::cpu_gemm_naive<T>(m, n, k, A.get_buf(), B.get_buf(), C.get_buf());
    return C;
  }

  template <typename T>
  inline auto cpu_srgemm_naive_entry(const Matrix<T> &A, const Matrix<T> &B)
      -> Matrix<T> {
    const auto m = A.num_rows();
    const auto k = A.num_cols(); // B.num_rows();
    const auto n = B.num_cols();
    auto C       = Matrix<T>(m, n);
    fwgpu::cpu_srgemm_naive(m, n, k, A.get_buf(), B.get_buf(), C.get_buf());
    return C;
  }
} // namespace testing
} // namespace fwgpu

#endif // GPU_GEMM_BM_ENTRY

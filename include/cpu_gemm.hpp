#ifndef FWGPU_CPU_GEMM_HPP
#define FWGPU_CPU_GEMM_HPP

#include <algorithm>

#include "Matrix.hpp"

namespace fwgpu {

template <typename T>
inline auto cpu_gemm_naive(int m, int n, int k, const T *A, const T *B, T *C) -> void {
  int lda = m;
  int ldb = k;
  int ldc = m;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      C[row + (col * ldc)] = 0.0;
      for (int i = 0; i < k; ++i) {
        C[row + (col * ldc)] += A[row + (i * lda)] * B[i + (col * ldb)];
      }
    }
  }
}

template <typename T>
inline auto naive_mm(const Matrix<T> &A, const Matrix<T> &B) -> Matrix<T> {
  const auto m = A.num_rows();
  const auto n = B.num_cols();
  const auto k = A.num_cols();
  auto c       = Matrix<T>(m, n);
  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < n; ++col) {
      c(row, col) = 0.0;
      for (size_t i = 0; i < k; ++i) {
        c(row, col) += A(row, i) * B(i, col);
      }
    }
  }
  return c;
}

} // namespace fwgpu

#endif // FWGPU_CPU_GEMM_HPP

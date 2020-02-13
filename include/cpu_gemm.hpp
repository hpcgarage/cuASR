#ifndef FWGPU_CPU_GEMM_HPP
#define FWGPU_CPU_GEMM_HPP

#include <algorithm>

#include "Matrix.hpp"

namespace fwgpu {

template <typename T>
auto simple_gemm(int m, int n, int k, const T *A, const T *B, T *C) -> void {
  for (auto i = 0; i < n; i++) {
    for (auto j = 0; j < n; j++) {
      for (auto k = 0; k < n; k++) {
        // column major indexing
        C[i + (n * j)] += A[i + (n * k)] * B[k + (n * j)];
      }
    }
  }
}

template <typename T>
auto simple_gemm_entry(const Matrix<T> &A, const Matrix<T> &B) -> Matrix<T> {
  const auto m = A.num_rows();
  const auto k = A.num_cols(); // B.num_rows();
  const auto n = B.num_cols();
  auto C       = Matrix<T>(m, n);
  simple_gemm(m, n, k, A.get_buf(), B.get_buf(), C.get_buf());
  return C;
}

template <typename T>
auto simple_srgemm(int m, int n, int k, const T *A, const T *B, T *C) -> void {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      T prod = A[i + (n * 0)] + B[0 + (n * j)];
      for (int k = 0; k < n; ++k) {
        prod = std::min(prod, A[i + (k * n)] + B[k + (j * n)]);
      }
      C[i * (n * j)] = prod;
    }
  }
}

template <typename T>
auto simple_srgemm_entry(const Matrix<T> &A, const Matrix<T> &B) -> Matrix<T> {
  const auto m = A.num_rows();
  const auto k = A.num_cols(); // B.num_rows();
  const auto n = B.num_cols();
  auto C       = Matrix<T>(m, n);
  simple_srgemm(m, n, k, A.get_buf(), B.get_buf(), C.get_buf());
  return C;
}

template <typename TData, typename TIdx>
auto simple_fwgemm(
    int m, int n, int k, const TData *A, const TData *B, TData *dist, TIdx *parent)
    -> void {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      // dist and parent for this vertex pair (i, j)
      TData curr_dist  = A[i + (n * 0)] + B[0 + (n * j)];
      TIdx curr_parent = 0;
      for (int k = 0; k < n; ++k) {
        TData prod = A[i + (k * n)] + B[k + (j * n)];
        if (prod < curr_dist) {
          curr_dist   = prod;
          curr_parent = k;
        }
      }
      dist[i * (n * j)]   = curr_dist;
      parent[i * (n * j)] = curr_parent;
    }
  }
}

template <typename T>
auto naive_mm(const Matrix<T> &A, const Matrix<T> &B) -> Matrix<T> {
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

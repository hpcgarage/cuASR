#ifndef FWGPU_CPU_SRGEMM_HPP
#define FWGPU_CPU_SRGEMM_HPP

namespace fwgpu {

template <typename T>
inline auto cpu_srgemm_naive(
    int m, int n, int k, const T *A, int lda, const T *B, int ldb, T *dist, int ldc)
    -> void {
  for (int kk = 0; kk < k; kk++) {
    for (int jj = 0; jj < n; jj++) {
      for (int ii = 0; ii < m; ii++) {
        if (dist[ii + (ldc * jj)] > A[ii + (lda * kk)] + B[kk + (ldb * jj)]) {
          dist[ii + (ldc * jj)] = A[ii + (lda * kk)] + B[kk + (ldb * jj)];
        }
      }
    }
  }
}

} // namespace fwgpu

#endif // FWGPU_CPU_SRGEMM_HPP

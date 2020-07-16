#ifndef FWGPU_CPU_FWGEMM_HPP
#define FWGPU_CPU_FWGEMM_HPP

namespace fwgpu {

template <typename T>
inline auto cpu_fwgemm_naive(
    int m,
    int n,
    int k,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    T *dist,
    int ldc,
    int *parent,
    int parent_offset) -> void {
  for (int kk = 0; kk < k; kk++) {
    for (int jj = 0; jj < n; jj++) {
      for (int ii = 0; ii < m; ii++) {
        if (dist[ii + (ldc * jj)] > A[ii + (lda * kk)] + B[kk + (ldb * jj)]) {
          dist[ii + (ldc * jj)]   = A[ii + (lda * kk)] + B[kk + (ldb * jj)];
          parent[ii + (ldc * jj)] = kk + parent_offset;
        }
      }
    }
  }
}

} // namespace fwgpu

#endif // FWGPU_CPU_FWGEMM_HPP

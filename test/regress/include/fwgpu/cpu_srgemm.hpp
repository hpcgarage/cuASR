#pragma once

namespace fwgpu {

template <typename T>
inline auto cpu_srgemm_naive(
    int m, int n, int k, const T *A, int lda, const T *B, int ldb, T *C, int ldc)
    -> void {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      T mindist = C[row + (col * ldc)];
      for (int i = 0; i < k; ++i) {
        mindist = std::min(mindist, A[row + (i * lda)] + B[i + (col * ldb)]);
      }
      C[row + (col * ldc)] = mindist;
    }
  }
}

template <typename TData, typename TIdx>
inline auto cpu_fwgemm_naive(
    int m,
    int n,
    int k,
    const TData *A,
    int lda,
    const TData *B,
    int ldb,
    TData *dist,
    int ldc,
    TIdx *parent) -> void {
  for (int row = 0; row < n; ++row) {
    for (int col = 0; col < n; ++col) {
      // dist and parent for this vertex pair (i, j)
      TData curr_dist  = dist[row + (col * ldc)];
      TIdx curr_parent = parent[row + (col * ldc)];
      for (int k = 0; k < n; ++k) {
        TData prod = A[row + (k * lda)] + B[k + (col * ldb)];
        if (prod < curr_dist) {
          curr_dist   = prod;
          curr_parent = k;
        }
      }
      dist[row + (col * ldc)]   = curr_dist;
      parent[row + (col * ldc)] = curr_parent;
    }
  }
}

} // namespace fwgpu

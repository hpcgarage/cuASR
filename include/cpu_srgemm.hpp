#include <limits>

namespace fwgpu {


template <typename T>
inline auto cpu_srgemm_naive(int m, int n, int k, const T *A, const T *B, T *C) -> void {
  int lda = m;
  int ldb = k;
  int ldc = m;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      T mindist = std::numeric_limits<T>::infinity();
      for (int i = 0; i < k; ++i) {
        mindist = std::min(mindist, A[row + (i * lda)] + B[i + (col * ldb)]);
      }
      C[row + (col * ldc)] = mindist;
    }
  }
}

template <typename TData, typename TIdx>
inline auto cpu_fwgemm_naive(
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
      dist[i + (n * j)]   = curr_dist;
      parent[i + (n * j)] = curr_parent;
    }
  }
}

} // namespace fwgpu

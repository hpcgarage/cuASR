#pragma once

#include <limits>

namespace fwgpu {

template <typename T>
__global__ auto
gpu_srgemm_naive(int m, int n, int k, T *A, int lda, T *B, int ldb, T *dist, int ldc)
    -> void {
  size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t n_idx = ty;
  while (n_idx < n) {
    size_t m_idx = tx;
    while (m_idx < m) {
      // initialize current minimum distance
      T mindist = dist[(n_idx * ldc) + m_idx];
      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        // calculate the distance between n_idx->m_idx by going through k_idx
        T thisone = A[(k_idx * lda) + m_idx] + B[(n_idx * ldb) + k_idx];
        if (thisone < mindist) {
          mindist = thisone;
        }
      }
      // finally, store new min distance to dist matrix
      dist[(n_idx * ldc) + m_idx] = mindist;
      m_idx += gridDim.x * blockDim.x;
    }
    n_idx += gridDim.y * blockDim.y;
  }
}

} // namespace fwgpu

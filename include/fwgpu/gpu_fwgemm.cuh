#ifndef FWGPU_GPU_FWGEMM_CUH
#define FWGPU_GPU_FWGEMM_CUH

#include <limits>

namespace fwgpu {

template <typename T>
__global__ auto gpu_fwgemm_naive(
    int m,
    int n,
    int k,
    T const *__restrict__ A,
    int lda,
    T const *__restrict__ B,
    int ldb,
    T *__restrict__ dist,
    int lddist,
    int *__restrict__ parent,
    int parent_offset,
    bool do_epilogue_min = true,
    void *stream         = nullptr) -> void {
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  int tx = blockIdx.x * blockDim.x + threadIdx.x;

  int n_idx = ty;
  while (n_idx < n) {
    int m_idx = tx;
    while (m_idx < m) {
      // initialize accumulators
      auto runnign_min_dist = std::numeric_limits<T>::infinity();
      int running_parent    = 0;
      if (do_epilogue_min) {
        runnign_min_dist = dist[(n_idx * lddist) + m_idx];
        running_parent   = parent[(n_idx * lddist) + m_idx];
      }

      // FW main loop
      for (int k_idx = 0; k_idx < k; ++k_idx) {
        // calculate the distance between n_idx->m_idx by going through k_idx
        auto curr_dist = A[(k_idx * lda) + m_idx] + B[(n_idx * ldb) + k_idx];
        if (curr_dist < runnign_min_dist) {
          runnign_min_dist = curr_dist;
          running_parent   = k_idx + parent_offset;
        }
      }

      // store final output
      dist[(n_idx * lddist) + m_idx]   = runnign_min_dist;
      parent[(n_idx * lddist) + m_idx] = running_parent;

      m_idx += gridDim.x * blockDim.x;
    }
    n_idx += gridDim.y * blockDim.y;
  }
}

} // namespace fwgpu

#endif // FWGPU_GPU_FWGEMM_HPP

#include <limits>

namespace fwgpu {

// TODO: here, left, right and dist can all be the same input matrix...
template <typename T>
__global__ auto gpu_srgemm_naive(int m, int n, int k, T *left, T *right, T *dist)
    -> void {
  size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t n_idx = ty;
  while (n_idx < n) {
    size_t m_idx = tx;
    while (m_idx < m) {
      // initialize current minimum distance
      T mindist = left[(0 * m) + m_idx] + right[(n_idx * k) + 0];
      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        // calculate the distance between n_idx->m_idx by going through k_idx
        T thisone = left[(k_idx * m) + m_idx] + right[(n_idx * k) + k_idx];
        if (thisone < mindist) {
          mindist = thisone;
        }
      }
      // finally, store min distance to dist matrix
      dist[(n_idx * m) + m_idx] = mindist;
      m_idx += gridDim.x * blockDim.x;
    }
    n_idx += gridDim.y * blockDim.y;
  }
}

} // namespace fwgpu

#ifndef FWGPU_GPU_GEMM_CUH
#define FWGPU_GPU_GEMM_CUH

#include "Matrix.hpp"

namespace fwgpu {
auto cublas_sgemm(
    const float *A, const float *B, float *C, const int m, const int k, const int n)
    -> void;

template <typename T>
__global__ auto gpu_gemm_naive(
    int m,
    int n,
    int k,
    const T *__restrict__ left,
    const T *__restrict__ right,
    T *__restrict__ dest) -> void {
  size_t ty = blockIdx.y * blockDim.y + threadIdx.y;
  size_t tx = blockIdx.x * blockDim.x + threadIdx.x;

  size_t n_idx = ty;
  while (n_idx < n) {
    size_t m_idx = tx;
    while (m_idx < m) {
      T tmp = static_cast<T>(0.0);
      for (size_t k_idx = 0; k_idx < k; ++k_idx) {
        tmp += left[(k_idx * m) + m_idx] * right[(n_idx * k) + k_idx];
      }
      dest[(n_idx * m) + m_idx] += tmp;
      m_idx += gridDim.x * blockDim.x;
    }
    n_idx += gridDim.y * blockDim.y;
  }
}

// clang-format off
// in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ auto gpu_gemm_sh_reg(
    int m,
    int n,
    int k,
    const T *__restrict__ lhs,
    const T *__restrict__ rhs,
    T *__restrict__ dest) -> void {
  using int_t = int; // either int or size_t
  __shared__ T lhs_shbuf[TILE_EXT_K][TILE_EXT_M];
  __shared__ T rhs_shbuf[TILE_EXT_N][TILE_EXT_K];

  for (int_t n_idx = blockIdx.y * TILE_EXT_N; n_idx < n; n_idx += gridDim.y * TILE_EXT_N) {
    // tile offset in Y dimension
    int_t n_end = n_idx + TILE_EXT_N;
    if (n_end > n) {
      n_end = n;
    }

    for (int_t m_idx = blockIdx.x * TILE_EXT_M; m_idx < m; m_idx += gridDim.x * TILE_EXT_M) {
      // tile offset in X dimension
      int_t m_end = m_idx + TILE_EXT_M;
      if (m_end > m) {
        m_end = m;
      }

      if ((m_end - m_idx == TILE_EXT_M) && (n_end - n_idx == TILE_EXT_N)) {
        // complete tile C(TILE_EXT_M,TILE_EXT_N)

        // Initialize registers to zero:
        T destreg[4][4] = { static_cast<T>(0.0) };
        T rhs_inreg[4]  = { static_cast<T>(0.0) };
        T lhs_inreg[4]  = { static_cast<T>(0.0) };

        for (int_t k_idx = 0; k_idx < k; k_idx += TILE_EXT_K) {
          // k_idx is the position of the CUDA thread along the K dimension
          int_t k_end = k_idx + TILE_EXT_K;
          if (k_end > k) {
            k_end = k;
          }

          // Load a tile of matrix A(m_idx:TILE_EXT_M, k_idx:TILE_EXT_K):
          for (int_t m_loc = m_idx + threadIdx.x; m_loc < m_end; m_loc += blockDim.x) {
            for (int_t k_loc = k_idx + threadIdx.y; k_loc < k_end; k_loc += blockDim.y) {
              lhs_shbuf[k_loc - k_idx][m_loc - m_idx] = lhs[k_loc * m + m_loc];
            }
          }

          // Load a tile of matrix B(k_idx:TILE_EXT_K, n_idx:TILE_EXT_N):
          for (int_t n_loc = n_idx + threadIdx.y; n_loc < n_end; n_loc += blockDim.y) {
            for (int_t k_loc = k_idx + threadIdx.x; k_loc < k_end; k_loc += blockDim.x) {
              rhs_shbuf[n_loc - n_idx][k_loc - k_idx] = rhs[n_loc * k + k_loc];
            }
          }

          // sync between load and use
          __syncthreads();

          // Multiply two loaded tiles to produce a tile of matrix
          // C(m_idx:TILE_EXT_M,n_idx:TILE_EXT_N):
          if (k_end - k_idx == TILE_EXT_K) {
            #pragma unroll
            for (int_t l = 0; l < TILE_EXT_K; ++l) {
              #pragma unroll
              for (int_t j = 0; j < 4; ++j) {
                rhs_inreg[j] = rhs_shbuf[threadIdx.y + blockDim.y * j][l];
              }
              #pragma unroll
              for (int_t j = 0; j < 4; ++j) {
                lhs_inreg[j] = lhs_shbuf[l][threadIdx.x + blockDim.x * j];
              }
              #pragma unroll
              for (int_t j = 0; j < 4; ++j) {
                #pragma unroll
                for (int_t i = 0; i < 4; ++i) {
                  destreg[j][i] += lhs_inreg[i] * rhs_inreg[j];
                }
              }
            }
          }
          else {
            for (int_t l = 0; l < (k_end - k_idx); ++l) {
              #pragma unroll
              for (int_t j = 0; j < 4; ++j) {
                rhs_inreg[j] = rhs_shbuf[threadIdx.y + blockDim.y * j][l];
              }

              #pragma unroll
              for (int_t j = 0; j < 4; ++j) {
                lhs_inreg[j] = lhs_shbuf[l][threadIdx.x + blockDim.x * j];
              }

              #pragma unroll
              for (int_t j = 0; j < 4; ++j) {
                #pragma unroll
                for (int_t i = 0; i < 4; ++i) {
                  destreg[j][i] += lhs_inreg[i] * rhs_inreg[j];
                }
              }
            }
          }

          // sync between each iteration of TILE_EXT_K
          __syncthreads();

        } // k_idx

        // Store elements of the C matrix in global memory
        #pragma unroll
        for (int_t j = 0; j < 4; ++j) {
          #pragma unroll
          for (int_t i = 0; i < 4; ++i) {
            size_t idx = (n_idx + threadIdx.y + blockDim.y * j) * m + (m_idx + threadIdx.x + blockDim.x * i);
            dest[idx] += destreg[j][i];
          }
        }
      }
      else {
        // incomplete tile of C
        // Initialize registers to zero:
        T destreg[4][4] = { static_cast<T>(0.0) };
        T rhs_inreg[4]  = { static_cast<T>(0.0) };
        T lhs_inreg[4]  = { static_cast<T>(0.0) };

        for (int_t k_idx = 0; k_idx < k; k_idx += TILE_EXT_K) {
          // k_idx is the position of the CUDA thread
          // along the K dimension
          int_t k_end = k_idx + TILE_EXT_K;
          if (k_end > k) {
            k_end = k;
          }

          // Load a tile of matrix A(m_idx:TILE_EXT_M, k_idx:TILE_EXT_K):
          for (int_t m_loc = m_idx + threadIdx.x; m_loc < m_end; m_loc += blockDim.x) {
            for (int_t k_loc = k_idx + threadIdx.y; k_loc < k_end; k_loc += blockDim.y) {
              lhs_shbuf[k_loc - k_idx][m_loc - m_idx] = lhs[k_loc * m + m_loc];
            }
          }

          // Load a tile of matrix B(k_idx:TILE_EXT_K, n_idx:TILE_EXT_N):
          for (int_t n_loc = n_idx + threadIdx.y; n_loc < n_end; n_loc += blockDim.y) {
            for (int_t k_loc = k_idx + threadIdx.x; k_loc < k_end; k_loc += blockDim.x) {
              rhs_shbuf[n_loc - n_idx][k_loc - k_idx] = rhs[n_loc * k + k_loc];
            }
          }

          // sync between load and use
          __syncthreads();

          // Multiply two loaded tiles to produce a tile of matrix
          // C(m_idx:TILE_EXT_M,n_idx:TILE_EXT_N):
          for (int_t l = 0; l < (k_end - k_idx); ++l) {
            for (int_t i = 0, j = threadIdx.y; j < n_end - n_idx; j += blockDim.y, i++) {
              rhs_inreg[i] = rhs_shbuf[j][l];
            }
            for (int_t i = 0, j = threadIdx.x; j < m_end - m_idx; j += blockDim.x, i++) {
              lhs_inreg[i] = lhs_shbuf[l][j];
            }
            #pragma unroll
            for (int_t j = 0; j < 4; ++j) {
              #pragma unroll
              for (int_t i = 0; i < 4; ++i) {
                destreg[j][i] += lhs_inreg[i] * rhs_inreg[j];
              }
            }
          }

          // sync between each iteration of TILE_EXT_K
          __syncthreads();

        } // k_idx

        // Store element of the C matrix in global memory:
        for (int_t j = 0, n_loc = n_idx + threadIdx.y; n_loc < n_end; n_loc += blockDim.y, j++) {
          for (int_t i = 0, m_loc = m_idx + threadIdx.x; m_loc < m_end; m_loc += blockDim.x, i++) {
            dest[n_loc * m + m_loc] += destreg[j][i];
          }
        }
      }
    } // m_idx
  } // n_idx
}
// clang-format on

} // namespace fwgpu

#endif // FWGPU_GPU_GEMM_CUH

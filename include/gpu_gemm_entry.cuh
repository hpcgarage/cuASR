#ifndef GPU_GEMM_BM_ENTRY
#define GPU_GEMM_BM_ENTRY

#include <chrono>
#include <tuple>

#include "Matrix.hpp"

namespace fwgpu {

auto cublas_sgemm_entry(const Matrix<float> &A, const Matrix<float> &B)
    -> Matrix<float>;

auto gpu_sgemm_sh_reg_entry(const Matrix<float> &A, const Matrix<float> &B)
    -> Matrix<float>;

auto gpu_sgemm_naive_entry(const Matrix<float> &A, const Matrix<float> &B)
    -> Matrix<float>;
} // namespace fwgpu

#endif // GPU_GEMM_BM_ENTRY

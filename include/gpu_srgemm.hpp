#ifndef FWGPU_GPU_GEMM_HPP
#define FWGPU_GPU_GEMM_HPP

namespace fwgpu {

auto cutlass_srsgemm_nn(
    int M,
    int N,
    int K,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float *C,
    int ldc,
    void *stream = nullptr) -> int;

} // namespace fwgpu
#endif

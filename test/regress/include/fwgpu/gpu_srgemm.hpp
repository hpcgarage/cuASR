#pragma once

namespace fwgpu {

// Cutlass semiring gemm based on {sum, min} as ring operators
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
    float *D,
    bool do_epilogue_min = true,
    void *stream         = nullptr) -> int;

// Cutlass semiring sgemm based on {sum, min} as ring operators
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
    bool do_epilogue_min = true,
    void *stream         = nullptr) -> int;

} // namespace fwgpu

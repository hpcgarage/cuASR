#include "cutlass/gemm/device/gemm.h"
namespace fwgpu {

cudaError_t cutlass_sgemm_nn(
    int M,
    int N,
    int K,
    float alpha,
    float const *A,
    int lda,
    float const *B,
    int ldb,
    float beta,
    float *C,
    int ldc,
    cudaStream_t stream = nullptr) {
  using ColumnMajor = cutlass::layout::ColumnMajor;
  using CutlassGemm = cutlass::gemm::device::Gemm<
      float,        // Data-type of A matrix
      ColumnMajor,  // Layout of A matrix
      float,        // Data-type of B matrix
      ColumnMajor,  // Layout of B matrix
      float,        // Data-type of C matrix
      ColumnMajor>; // Layout of C matrix

  CutlassGemm::Arguments args(
      { M, N, K }, // Gemm Problem dimensions
      { A, lda },  // Tensor-ref for source matrix A
      { B, ldb },  // Tensor-ref for source matrix B
      { C, ldc },  // Tensor-ref for source matrix C
      { C, ldc },  // Tensor-ref for destination matrix D (may be different memory than
                   // source C matrix)
      { alpha, beta }); // Scalars used in the Epilogue

  CutlassGemm gemm_operator;
  cutlass::Status status = gemm_operator(args, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

} // namespace fwgpu

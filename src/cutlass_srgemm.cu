#include "fwgpu/gpu_srgemm.hpp"

#include "cutlass/gemm/device/default_srgemm_configuration.h"
#include "cutlass/gemm/device/srgemm.h"

#include <cuda_runtime.h>

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
    float *D,
    bool do_epilogue_min,
    void *stream) -> int {
  cudaStream_t stream_ = nullptr;
  if (stream) {
    stream_ = *(static_cast<cudaStream_t *>(stream));
  }
  // compile time configuration of this srgemm kernel
  using OperatorClass = cutlass::arch::OpClassSimt;
  using ArchTag       = cutlass::arch::Sm50;
  using DefaultConfig = typename cutlass::gemm::device::DefaultSrgemmConfiguration<
      OperatorClass, ArchTag, float, float, float, float>;

  using ColumnMajor        = cutlass::layout::ColumnMajor;
  using ThreadblockShape   = typename DefaultConfig::ThreadblockShape;
  using WarpShape          = typename DefaultConfig::WarpShape;
  using InstructionShape   = typename DefaultConfig::InstructionShape;
  using EpilogueOutputOp   = typename DefaultConfig::EpilogueOutputOp;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  using Operator           = DefaultConfig::Operator;

  using CutlassSrgemm = cutlass::gemm::device::Srgemm<
      float,                      // element type of A
      ColumnMajor,                // layout of A
      float,                      // element type of B
      ColumnMajor,                // layout of B
      float,                      // element type of C
      ColumnMajor,                // layout of C
      float,                      // element type of D
      OperatorClass,              // Logical operator class (SIMT/Tensor)
      ArchTag,                    // cuda architecture
      ThreadblockShape,           // GEMM shape at CTA level
      WarpShape,                  // GEMM shape at Warp level
      InstructionShape,           // GEMM shape at thread level
      EpilogueOutputOp,           // Epilogue operator at thread level
      ThreadblockSwizzle,         // GEMM threadblock swizzler
      DefaultConfig::kStages,     // Pipeline stages for shmem
      DefaultConfig::kAlignmentA, // Alignment of A elements
      DefaultConfig::kAlignmentB, // Alignment of B elements
      false,                      // SplitKSerial
      Operator                    // Thread level SemiRing operator
      >;

  // construct kernel arguments struct
  CutlassSrgemm::Arguments args(
      { M, N, K },        // Problem dimensions
      { A, lda },         // Tensor-ref for source matrix A
      { B, ldb },         // Tensor-ref for source matrix B
      { C, ldc },         // Tensor-ref for source matrix C
      { D, ldc },         // Tensor-ref for destination matrix D
      { do_epilogue_min } // True if we perform a final min with source matrix C
  );

  // launch SRGEMM kernel
  CutlassSrgemm srgemm_operator;
  cutlass::Status status = srgemm_operator(args, nullptr, stream_);
  return static_cast<int>(status);
}

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
    bool do_epilogue_min,
    void *stream) -> int {
  return cutlass_srsgemm_nn(M, N, K, A, lda, B, ldb, C, ldc, C, do_epilogue_min, stream);
}

} // namespace fwgpu

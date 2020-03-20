#include "include/gpu_srgemm.hpp"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm.h"

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
    void *stream_) -> int {
  auto stream         = static_cast<cudaStream_t>(stream_);
  using ColumnMajor   = cutlass::layout::ColumnMajor;
  using OperatorClass = cutlass::arch::OpClassSimt;
  using ArchTag       = cutlass::arch::Sm50;

  using DefaultConfig = typename cutlass::gemm::device::DefaultSrgemmConfiguration<
      OperatorClass, ArchTag, float, float, float, float>;

  using ThreadblockShape   = typename DefaultConfig::ThreadblockShape;
  using WarpShape          = typename DefaultConfig::WarpShape;
  using InstructionShape   = typename DefaultConfig::InstructionShape;
  using EpilogueOutputOp   = typename DefaultConfig::EpilogueOutputOp;
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle;
  using Operator           = DefaultConfig::Operator;

  using CutlassSrgemm = cutlass::gemm::device::Srgemm<
      float,                                 //
      ColumnMajor,                           //
      float,                                 //
      ColumnMajor,                           //
      float,                                 //
      ColumnMajor,                           //
      float,                                 //
      OperatorClass,                         //
      ArchTag,                               //
      cutlass::gemm::GemmShape<128, 128, 8>, //
      WarpShape,                             //
      InstructionShape,                      //
      EpilogueOutputOp,                      //
      ThreadblockSwizzle,                    //
      DefaultConfig::kStages,                //
      DefaultConfig::kAlignmentA,            //
      DefaultConfig::kAlignmentB,            //
      false,                                 // SplitKSerial
      Operator
      // true                                // IsBetaZero
      >; // Layout of C matrix

  CutlassSrgemm srgemm_operator;

  CutlassSrgemm::Arguments args(
      { M, N, K }, // Problem dimensions
      { A, lda },  // Tensor-ref for source matrix A
      { B, ldb },  // Tensor-ref for source matrix B
      { C, ldc },  // Tensor-ref for source matrix C
      { C, ldc },  // Tensor-ref for destination matrix D (may be different memory than
                   // source C matrix)
      { 1, 0 });   // Scalars used in the Epilogue

  // launch SRGEMM kernel
  cutlass::Status status = srgemm_operator(args, nullptr, stream);

  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(cudaErrorUnknown);
  }
  return static_cast<int>(cudaSuccess);
}

} // namespace fwgpu

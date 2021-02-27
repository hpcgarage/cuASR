#include "fwgpu/gpu_srgemm.hpp"

#include "cuasr/arch/srmma.h"
#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"

#include "cutlass/functional.h"

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
  using OperatorClass  = cutlass::arch::OpClassSimt;
  using SmArch         = cutlass::arch::Sm50;
  using TropicalConfig = typename cuasr::gemm::device::DefaultSemiRingConfiguration<
      float, float, float, float, OperatorClass, cuasr::minimum<float>,
      cuasr::plus<float>, SmArch>;

  using AdditionOp       = TropicalConfig::AdditionOp;
  using MultiplicationOp = TropicalConfig::MultiplicationOp;
  using ColumnMajor      = cutlass::layout::ColumnMajor;
  using ThreadblockShape = typename TropicalConfig::ThreadblockShape;
  using WarpShape        = typename TropicalConfig::WarpShape;
  using InstructionShape = typename TropicalConfig::InstructionShape;
  using EpilogueOutputOp = typename TropicalConfig::EpilogueOutputOp;
  using ThreadblockSwizzle =
      typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  constexpr int Stages     = TropicalConfig::kStages;
  constexpr int AlignmentA = TropicalConfig::kAlignmentA;
  constexpr int AlignmentB = TropicalConfig::kAlignmentB;

  using cuASR_MinPlus_SGEMM = cuasr::gemm::device::Srgemm<
      AdditionOp,         // Thread level SemiRing operator
      MultiplicationOp,   // Thread level SemiRing operator
      float,              // element type of A
      ColumnMajor,        // layout of A
      float,              // element type of B
      ColumnMajor,        // layout of B
      float,              // element type of C
      ColumnMajor,        // layout of C
      float,              // element type of D
      OperatorClass,      // Logical operator class (SIMT/Tensor)
      SmArch,             // cuda architecture
      ThreadblockShape,   // GEMM shape at CTA level
      WarpShape,          // GEMM shape at Warp level
      InstructionShape,   // GEMM shape at thread level
      EpilogueOutputOp,   // Epilogue operator at thread level
      ThreadblockSwizzle, // GEMM threadblock swizzler
      Stages,             // Pipeline stages for shmem
      AlignmentA,         // Alignment of A elements
      AlignmentB,         // Alignment of B elements
      false               // SplitKSerial
  >;

  float alpha = MultiplicationOp::Identity;
  float beta
      = do_epilogue_min ? MultiplicationOp::Identity : MultiplicationOp::Annihilator;
  // construct kernel arguments struct
  cuASR_MinPlus_SGEMM::Arguments args(
      { M, N, K },    // Problem dimensions
      { A, lda },     // Tensor-ref for source matrix A
      { B, ldb },     // Tensor-ref for source matrix B
      { C, ldc },     // Tensor-ref for source matrix C
      { D, ldc },     // Tensor-ref for destination matrix D
      { alpha, beta } // True if we perform a final min with source matrix C
  );

  // launch SRGEMM kernel
  cuASR_MinPlus_SGEMM minplus_gemm;
  cutlass::Status status = minplus_gemm(args, nullptr, stream_);
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

/***************************************************************************************************

 **************************************************************************************************/

/*! \file
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cuasr/gemm/kernel/default_srgemm.h"
#include "cuasr/gemm/kernel/srgemm_splitk_parallel.h"

////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Operator class tag
  typename OperatorClass,
  /// Tag indicating architecture to tune for
  typename ArchTag,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  /// Addition operator of the semi-ring
  typename AdditionOp,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// Number of stages used in the pipelined mainloop
  int Stages
>
struct DefaultSrgemmSplitKParallel {

  // Define threadblock-scoped split-K matrix multiply using
  // the basic SRGEMM's kernel level main loop
  using Default = DefaultSrgemm<
    ElementA_,
    LayoutA_,
    kAlignmentA,
    ElementB_,
    LayoutB_,
    kAlignmentB,
    ElementAccumulator,
    LayoutC_,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    AdditionOp,
    MultiplicationOp,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    false
  >;

  /// Define the semiring matrix multiply operator
  using Srmma = typename Default::Srmma;

  /// Define the epilogue
  using Epilogue = typename Default::Epilogue;

  /// Define the kernel-level GEMM operator.
  using SrgemmKernel = kernel::SrgemmSplitKParallel<
      Srmma,
      AdditionOp,
      MultiplicationOp,
      Epilogue,
      ThreadblockSwizzle
  >;
};

///////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cuasr

///////////////////////////////////////////////////////////////////////////////////////////////////

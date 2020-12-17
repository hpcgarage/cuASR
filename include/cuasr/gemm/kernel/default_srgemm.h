/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief
      Default kernel-level SRGEMM definitions combine threadblock-scoped matrix srmma
      with the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "cuasr/arch/srmma.h"
#include "cuasr/gemm/kernel/srgemm.h"
#include "cuasr/gemm/threadblock/default_srmma.h"

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
    /// Instruction-level tile size (concept: GemmShape)
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
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial>
struct DefaultSrgemm;

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Addition operator of the semi-ring
    typename AdditionOp,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial
>
struct DefaultSrgemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    cutlass::gemm::GemmShape<1, 1, 1>,
    AdditionOp,
    MultiplicationOp,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    2,
    SplitKSerial> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Srmma = typename cuasr::gemm::threadblock::DefaultSrmma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm50,
      ThreadblockShape,
      WarpShape,
      cutlass::gemm::GemmShape<1, 1, 1>,
      AdditionOp,
      MultiplicationOp,
      2>::ThreadblockSrmma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Srmma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using SrgemmKernel = cuasr::gemm::kernel::Srgemm<
      Srmma,
      AdditionOp,
      MultiplicationOp,
      Epilogue,
      ThreadblockSwizzle,
      SplitKSerial
  >;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cuasr

////////////////////////////////////////////////////////////////////////////////

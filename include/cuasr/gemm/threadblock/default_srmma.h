/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

#include "cuasr/gemm/threadblock/default_srmma_core.h"

////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace threadblock {

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
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Store the accumulators in row major or column major.
    /// Row major is used when output layout is interleaved.
    bool AccumulatorsInRowMajor = false
>
struct DefaultSrmma;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass Simt)
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
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
    typename MultiplicationOp>
struct DefaultSrmma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, cutlass::layout::RowMajor,
                  cutlass::arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, AdditionOp, MultiplicationOp, 2, false> {
  // Define the SrmmaCore components
  using SrmmaCore = typename cuasr::gemm::threadblock::DefaultSrmmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, cutlass::layout::RowMajor,
      cutlass::arch::OpClassSimt, AdditionOp, MultiplicationOp, 2>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<SrmmaCore::Shape::kM, SrmmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename SrmmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<SrmmaCore::Shape::kK, SrmmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename SrmmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockSrmma = cuasr::gemm::threadblock::SrmmaPipelined<
      typename SrmmaCore::Shape, IteratorA, typename SrmmaCore::SmemIteratorA,
      IteratorB, typename SrmmaCore::SmemIteratorB, ElementAccumulator,
      cutlass::layout::RowMajor, typename SrmmaCore::MmaPolicy>;
};


} // namespace threadblock
} // namespace gemm
} // namespace cuasr

////////////////////////////////////////////////////////////////////////////////

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

#include "fwgpu/srgemm/threadblock/default_srmma_core.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
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
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
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
    /// Operation performed by GEMM
    typename Operator>
struct DefaultSrmma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, 2, Operator, false> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultSrmmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassSimt, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::SrmmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};


} // namespace threadblock
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

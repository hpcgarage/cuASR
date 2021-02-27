/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Defines basic properties needed by CTA-level GEMMs assuming expectations about data
      layout of the global memory fragments, data types, and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting TensorOp instructions.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/cache_operation.h"
#include "cutlass/gemm/warp/mma.h"

#include "cuasr/gemm/threadblock/srmma_pipelined.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Template defininng default matrix multiply operators inferred from threadblock tile size,
/// global memory data layout, and target math instruction.
template <
    /// Shape of threadblock-scoped matrix multiply operator
    typename Shape,
    /// Shape of warp-level matrix multiply operator
    typename WarpShape,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape,
    /// Element data type of A operand
    typename ElementA,
    /// Layout of operand A
    typename LayoutA,
    /// Element data type of B operand
    typename ElementB,
    /// Layout of operand B
    typename LayoutB,
    /// Data type of accumulator
    typename ElementC,
    /// Layout of accumulator
    typename LayoutC,
    /// Indicates type of math operator (arch::OpClassSimt or arch::OpClassTensorOp)
    typename OperatorClass,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_,
    /// Number of stages
    int Stages = 2,
    /// Store the accumulators in row major or column major.
    /// Row major is usedd when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA =
        cutlass::arch::CacheOperation::Global,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB =
        cutlass::arch::CacheOperation::Global,
    /// per-element transformation for elements of A
    cutlass::ComplexTransform TransformA = cutlass::ComplexTransform::kNone,
    /// per-element transformation for elements of B
    cutlass::ComplexTransform TransformB = cutlass::ComplexTransform::kNone,
    bool IsComplex = false // (is_complex<ElementA>::value || is_complex<ElementB>::value)
>
struct DefaultSrmmaCore;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cuasr

#include "cuasr/gemm/threadblock/default_srmma_core_simt.h"

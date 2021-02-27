/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Templates exposing architecture support for warp-level multiply-add operations
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/mma.h"

#include "cutlass/gemm/thread/mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape,
  /// Data type of A elements
  typename ElementA,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA,
  /// Data type of B elements
  typename ElementB,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB,
  /// Element type of C matrix
  typename ElementC,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC,
  /// Addition operator of the semi-ring
  typename AdditionOp,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp,
  /// Used for partial specialization
  typename Enable = bool
>
struct Srmma;

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace gemm
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////

//
// Overloads specialized for existing architectures
//

#include "cuasr/gemm/thread/srmma_sm50.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

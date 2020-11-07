/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace arch {

/// Matrix product operator for all semi-rings
template <
  /// Size of the matrix product (concept: GemmShape)
  typename Shape_,
  /// Number of threads participating
  int kThreads_,
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
  /// addition operator of the semi-ring
  typename AdditionOp,
  /// multiplication operator of the semi-ring
  typename MultiplicationOp
>
struct Srmma;


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Semi-rings multiply-add specialized for 1 element per instruction
template <
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
    typename MultiplicationOp>
struct Srmma<
    cutlass::gemm::GemmShape<1, 1, 1>,
    1,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    AdditionOp,
    MultiplicationOp> {
  using Shape = cutlass::gemm::GemmShape<1, 1, 1>;

  // semi-ring operators must be default contructible and
  // have a binary invocation () operator
  AdditionOp add;
  MultiplicationOp mult;

  CUTLASS_HOST_DEVICE
  void operator()(
    cutlass::Array<ElementC, 1> &d,
    cutlass::Array<ElementA, 1> const &a,
    cutlass::Array<ElementB, 1> const &b,
    cutlass::Array<ElementC, 1> const &c
  ) {
    d[0] = add(c[0], mult(a[0], b[0]));
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////

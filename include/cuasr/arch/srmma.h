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
  /// Ring operator that performa FMA
  typename RingOp
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
    /// Ring operator that performa FMA
    typename RingOp>
struct Srmma<
    cutlass::gemm::GemmShape<1, 1, 1>,
    1,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    RingOp> {
  using Shape = cutlass::gemm::GemmShape<1, 1, 1>;

  // semi-ring operators must be default contructible and
  // have a binary invocation () operator
  RingOp ring_op;

  CUTLASS_HOST_DEVICE
  void operator()(
    cutlass::Array<ElementC, 1> &d,
    cutlass::Array<ElementA, 1> const &a,
    cutlass::Array<ElementB, 1> const &b,
    cutlass::Array<ElementC, 1> const &c
  ) {
    ring_op.fma(d[0], a[0], b[0], c[0]);
  }
};

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
    typename LayoutC>
struct Srmma<
    cutlass::gemm::GemmShape<1, 1, 1>,
    1,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    min_plus<ElementA, 1>> {
  using Shape = cutlass::gemm::GemmShape<1, 1, 1>;

  // semi-ring operators must be default contructible and
  // have a binary invocation () operator
  min_plus<ElementA, 1> ring_op;

  CUTLASS_HOST_DEVICE
  void operator()(
    cutlass::Array<ElementC, 1> &d,
    cutlass::Array<ElementA, 1> const &a,
    cutlass::Array<ElementB, 1> const &b,
    cutlass::Array<ElementC, 1> const &c
  ) {
    asm volatile (
    "\n{"
    "\n\t .reg .f32 r0;"
    "\n\t .reg .f32 r1;"
    "\n\t .reg .f32 r2;"
    "\n\t .reg .f32 r3;"
    "\n\t mov.f32 r1, %1;"
    "\n\t mov.f32 r2, %2;"
    "\n\t mov.f32 r3, %3;"
    "\n\t min.f32 r3, r1, r2;"
    "\n\t mov.f32 %0, r0;"
    "\n}"
    : "=f"(d[0])
    :  "f"(a[0]), "f"(b[0]), "f"(c[0])
    );
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////

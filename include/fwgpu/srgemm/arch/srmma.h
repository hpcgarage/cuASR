/***************************************************************************************************

 **************************************************************************************************/
/*! \file
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/mma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/// Tag indicating min-plus semi-ring (tropical semi-ring)
struct OpSumMin;

/////////////////////////////////////////////////////////////////////////////////////////////////

/// MMA specialization for tropical semiring d = min(c, a+b)
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
  typename LayoutC
>
struct Mma<gemm::GemmShape<1, 1, 1>, 1, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, OpSumMin> {

  using Shape = gemm::GemmShape<1, 1, 1>;

  CUTLASS_HOST_DEVICE
  void operator()(
    Array<ElementC, 1> &d,
    Array<ElementA, 1> const &a,
    Array<ElementB, 1> const &b,
    Array<ElementC, 1> const &c
  ) {
    d[0] = ((a[0] + b[0]) < c[0])
         ?  (a[0] + b[0])
         : c[0];
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

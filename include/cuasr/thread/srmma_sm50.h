/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

#include "cuasr/arch/srmma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles all packed matrix layouts
template <
  /// Size of the Gemm problem - concept: cutlass::gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: layout::MapFunc)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: layout::MapFunc)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: layout::MapFunc)
  typename LayoutC_,
  /// Addition operator of the semi-ring
  typename AdditionOp_,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp_
>
struct SrmmaGeneric {

  /// Size of the Gemm problem - concept: cutlass::gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = LayoutA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = LayoutB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// A operand storage
  using FragmentA = cutlass::Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = cutlass::Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = cutlass::Array<ElementC, Shape::kMN>;

  /// Instruction
  using SrmmaOp = arch::Srmma<
    cutlass::gemm::GemmShape<1,1,1>,
    1,
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    AdditionOp, MultiplicationOp>;

  //
  // Methods
  //

  /// Computes a generalized matrix product on any semi-ring
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    cutlass::TensorRef<ElementA const, LayoutA> a_ref(
      reinterpret_cast<ElementA const *>(&A), LayoutA::packed({Shape::kM, Shape::kK}));

    cutlass::TensorRef<ElementB const, LayoutB> b_ref(
      reinterpret_cast<ElementB const *>(&B), LayoutB::packed({Shape::kK, Shape::kN}));

    cutlass::TensorRef<ElementC, LayoutC> d_ref(
      reinterpret_cast<ElementC *>(&D), LayoutC::packed({ Shape::kM, Shape::kN }));

    SrmmaOp srmma_op;

    // Copy accumulators
    D = C;

    // Compute matrix product
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK; ++k) {

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < Shape::kN; ++n) {

        CUTLASS_PRAGMA_UNROLL
        for (int m = 0; m < Shape::kM; ++m) {

          int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;

          cutlass::MatrixCoord mn(m_serpentine, n);
          cutlass::MatrixCoord mk(m_serpentine, k);
          cutlass::MatrixCoord kn(k, n);

          cutlass::Array<ElementC, 1> d;
          cutlass::Array<ElementA, 1> a;
          cutlass::Array<ElementB, 1> b;

          d[0] = d_ref.at(mn);
          a[0] = a_ref.at(mk);
          b[0] = b_ref.at(kn);

          srmma_op(d, a, b, d);

          d_ref.at(mn) = d[0];
        }
      }
    }
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles conventional layouts for FFMA and DFMA GEMM
template <
  /// Size of the Gemm problem - concept: cutlass::gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: layout::MapFunc)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: layout::MapFunc)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: layout::MapFunc)
  typename LayoutC_,
  /// Addition operator of the semi-ring
  typename AdditionOp_,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp_
>
struct Srmma<
  Shape_,
  ElementA_,
  LayoutA_,
  ElementB_,
  LayoutB_,
  ElementC_,
  LayoutC_,
  AdditionOp_,
  MultiplicationOp_,
  bool
> {

  /// Size of the Gemm problem - concept: cutlass::gemm::GemmShape<>
  using Shape = Shape_;

  /// Data type of operand A
  using ElementA = ElementA_;

  /// Layout of A matrix (concept: layout::MapFunc)
  using LayoutA = LayoutA_;

  /// Data type of operand B
  using ElementB = ElementB_;

  /// Layout of B matrix (concept: layout::MapFunc)
  using LayoutB = LayoutB_;

  /// Element type of operand C
  using ElementC = ElementC_;

  /// Layout of C matrix (concept: layout::MapFunc)
  using LayoutC = LayoutC_;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  //
  // Methods
  //

  /// Computes a matrix product for any semi-ring
  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC & D,
    FragmentA const & A,
    FragmentB const & B,
    FragmentC const & C) {

    SrmmaGeneric<
      Shape,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      AdditionOp,
      MultiplicationOp> srmma;

    srmma(D, A, B, C);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace gemm
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////

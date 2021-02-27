/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Templates implementing warp-level matrix multiply-accumulate operations.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"
#include "cutlass/gemm/warp/mma_simt_tile_iterator.h"

#include "cuasr/gemm/thread/srmma.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Shape of the warp in units of thread (concept: MmaSimtPolicy)
  typename Policy_,
  /// Addition of the semi-ring
  typename AdditionOp_,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp_,
  /// Number of partitions along K dimension
  int PartitionsK = 1,
  /// Used for partial specialization
  typename Enable = bool
>
class SrmmaSimt {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Indicates class of matrix operator
  using OperatorClass = cutlass::arch::OpClassSimt;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  using ThreadLayoutA = typename cutlass::platform::conditional<
      cutlass::platform::is_same<cutlass::layout::ColumnMajorInterleaved<4>, LayoutA>::
          value,
      cutlass::layout::ColumnMajor,
      typename cutlass::platform::conditional<
          cutlass::platform::is_same<cutlass::layout::RowMajorInterleaved<4>, LayoutA>::
              value,
          cutlass::layout::RowMajor,
          LayoutA>::type>::type;

  using ThreadLayoutB = typename cutlass::platform::conditional<
      cutlass::platform::is_same<cutlass::layout::ColumnMajorInterleaved<4>, LayoutB>::
          value,
      cutlass::layout::ColumnMajor,
      typename cutlass::platform::conditional<
          cutlass::platform::is_same<cutlass::layout::RowMajorInterleaved<4>, LayoutB>::
              value,
          cutlass::layout::RowMajor,
          LayoutB>::type>::type;

  static constexpr bool use_dp4a
      = (cutlass::platform::is_same<cutlass::layout::ColumnMajorInterleaved<4>, LayoutA>::
             value
         || cutlass::platform::is_same<cutlass::layout::RowMajorInterleaved<4>, LayoutA>::
             value)
      && cutlass::platform::is_same<ElementA, int8_t>::value
      && cutlass::platform::is_same<ElementB, int8_t>::value;

  using dp4a_type = typename cutlass::platform::conditional< use_dp4a , int8_t, bool >::type;

  /// Thread-level matrix multiply accumulate operator
  using ThreadMma = cuasr::gemm::thread::Srmma<
    cutlass::gemm::GemmShape<
      Shape::kM / Policy::WarpShape::kRow,
      Shape::kN / Policy::WarpShape::kColumn,
      Policy::LaneMmaShape::kK>,
    ElementA,
    ThreadLayoutA,
    ElementB,
    ThreadLayoutB,
    ElementC,
    LayoutC,
    AdditionOp,
    MultiplicationOp,
    dp4a_type
  >;

public:

  /// Iterates over the A operand in memory
  using IteratorA = cutlass::gemm::warp::MmaSimtTileIterator<
    cutlass::MatrixShape<Shape::kM, Policy::LaneMmaShape::kK>,
    cutlass::gemm::Operand::kA,
    ElementA,
    LayoutA,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Iterates over the B operand in memory
  using IteratorB = cutlass::gemm::warp::MmaSimtTileIterator<
    cutlass::MatrixShape<Policy::LaneMmaShape::kK, Shape::kN>,
    cutlass::gemm::Operand::kB,
    ElementB,
    LayoutB,
    Policy,
    PartitionsK,
    Shape::kK
  >;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Iterates over the C operand in memory
  using IteratorC = cutlass::gemm::warp::MmaSimtTileIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kN>,
    cutlass::gemm::Operand::kC,
    ElementC,
    LayoutC,
    Policy
  >;

  /// Storage for C tile
  using FragmentC = typename ThreadMma::FragmentC;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  SrmmaSimt() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c, int group_idx = 0) const {

    ThreadMma srmma;

    srmma(d, a, b, c);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cuasr

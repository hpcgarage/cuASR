/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Definitions for SRGEMM configuration structures.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"

#include "cuasr/functional.h"
#include "cuasr/arch/srmma.h"
#include "cuasr/gemm/epilogue/thread/semiring_linear_combination.h"

#include <limits>

////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename AdditionOp,
  typename MultiplicationOp,
  typename ArchTag
>
struct DefaultSemiRingConfiguration;

////////////////////////////////////////////////////////////////////////////////

// Plus-Times semi-ring GEMM configuration
// this is the traditional GEMM
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::plus<Element>,
  cuasr::multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::plus<Element>;
  using MultiplicationOp = cuasr::multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

// Min-Plus (tropical) semi-ring GEMM configuration
// example application: All Pairs Shorted Path
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::minimum<Element>,
  cuasr::plus<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::minimum<Element>;
  using MultiplicationOp = cuasr::plus<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

// Max-Plus semi-ring GEMM configuration
// example application: Viterbi algorithm
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::maximum<Element>,
  cuasr::plus<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::maximum<Element>;
  using MultiplicationOp = cuasr::plus<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

// Max-Min
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::maximum<Element>,
  cuasr::minimum<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::maximum<Element>;
  using MultiplicationOp = cuasr::minimum<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

// Min-Max
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::minimum<Element>,
  cuasr::maximum<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::minimum<Element>;
  using MultiplicationOp = cuasr::maximum<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

// Min-Times
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::minimum<Element>,
  cuasr::multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::minimum<Element>;
  using MultiplicationOp = cuasr::multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

// Max-Times
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::maximum<Element>,
  cuasr::multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::maximum<Element>;
  using MultiplicationOp = cuasr::multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

// Or-And boolean ring
template <
  typename Element,
  typename ArchTag
>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  cutlass::arch::OpClassSimt,
  cuasr::binary_or<Element>,
  cuasr::binary_and<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cuasr::binary_or<Element>;
  using MultiplicationOp = cuasr::binary_and<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    AdditionOp, MultiplicationOp, Element, 1>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cuasr

////////////////////////////////////////////////////////////////////////////////

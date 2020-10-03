/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Definitions for SRGEMM configuration structures.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"

#include "cuasr/functional.h"
#include "cuasr/arch/srmma.h"
#include "cuasr/epilogue/thread/semiring_addition_op.h"

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
  cutlass::plus<Element>,
  cutlass::multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cutlass::plus<Element>;
  static Element constexpr AdditiveIdentity
      = static_cast<Element>(0);

  using MultiplicationOp = cutlass::multiplies<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(1);

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    cutlass::plus<cutlass::Array<Element, 1>>, Element, 1>;
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
  cutlass::minimum<Element>,
  cutlass::plus<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cutlass::minimum<Element>;
  static Element constexpr AdditiveIdentity
      = ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = cutlass::plus<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(0);

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    cutlass::minimum<cutlass::Array<Element, 1>>, Element, 1>;
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
  cutlass::maximum<Element>,
  cutlass::plus<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cutlass::maximum<Element>;
  static Element constexpr AdditiveIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = cutlass::plus<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(0);

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    cutlass::maximum<cutlass::Array<Element, 1>>, Element, 1>;
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
  cutlass::maximum<Element>,
  cutlass::minimum<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cutlass::maximum<Element>;
  static Element constexpr AdditiveIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = cutlass::minimum<Element>;
  static Element constexpr MultiplicativeIdentity
      = ::std::numeric_limits<Element>::infinity();

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    cutlass::maximum<cutlass::Array<Element, 1>>, Element, 1>;
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
  cutlass::minimum<Element>,
  cutlass::maximum<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cutlass::minimum<Element>;
  static Element constexpr AdditiveIdentity
      = ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = cutlass::maximum<Element>;
  static Element constexpr MultiplicativeIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    cutlass::minimum<cutlass::Array<Element, 1>>, Element, 1>;
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
  cutlass::minimum<Element>,
  cutlass::multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cutlass::minimum<Element>;
  static Element constexpr AdditiveIdentity
      = ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = cutlass::multiplies<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(1);

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    cutlass::minimum<cutlass::Array<Element, 1>>, Element, 1>;
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
  cutlass::maximum<Element>,
  cutlass::multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = cutlass::maximum<Element>;
  static Element constexpr AdditiveIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = cutlass::multiplies<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(1);

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    cutlass::maximum<cutlass::Array<Element, 1>>, Element, 1>;
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
  binary_or<Element>,
  binary_and<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = binary_or<Element>;
  static Element constexpr AdditiveIdentity
      = static_cast<Element>(false);

  using MultiplicationOp = binary_and<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(true);

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringAdditionOp<
    binary_or<cutlass::Array<Element, 1>>, Element, 1>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cuasr

////////////////////////////////////////////////////////////////////////////////

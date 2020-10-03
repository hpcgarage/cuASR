/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Definitions for SRGEMM configuration structures.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/functional.h"
#include "cuasr/functional.h"

#include "cuasr/arch/srmma.h"
#include "cutlass/gemm/gemm.h"

#include "cuasr/epilogue/thread/semiring_addition_op.h"

#include <limits>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
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
  arch::OpClassSimt,
  plus<Element>,
  multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<32, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = plus<Element>;
  static Element constexpr AdditiveIdentity
      = static_cast<Element>(0);

  using MultiplicationOp = multiplies<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(1);

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    plus<Array<Element, 1>>, Element, 1>;
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
  arch::OpClassSimt,
  minimum<Element>,
  plus<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = minimum<Element>;
  static Element constexpr AdditiveIdentity
      = ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = plus<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(0);

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    minimum<Array<Element, 1>>, Element, 1>;
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
  arch::OpClassSimt,
  maximum<Element>,
  plus<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = maximum<Element>;
  static Element constexpr AdditiveIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = plus<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(0);

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    maximum<Array<Element, 1>>, Element, 1>;
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
  arch::OpClassSimt,
  maximum<Element>,
  minimum<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = maximum<Element>;
  static Element constexpr AdditiveIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = minimum<Element>;
  static Element constexpr MultiplicativeIdentity
      = ::std::numeric_limits<Element>::infinity();

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    maximum<Array<Element, 1>>, Element, 1>;
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
  arch::OpClassSimt,
  minimum<Element>,
  maximum<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = minimum<Element>;
  static Element constexpr AdditiveIdentity
      = ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = maximum<Element>;
  static Element constexpr MultiplicativeIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    minimum<Array<Element, 1>>, Element, 1>;
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
  arch::OpClassSimt,
  minimum<Element>,
  multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = minimum<Element>;
  static Element constexpr AdditiveIdentity
      = ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = multiplies<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(1);

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    minimum<Array<Element, 1>>, Element, 1>;
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
  arch::OpClassSimt,
  maximum<Element>,
  multiplies<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = maximum<Element>;
  static Element constexpr AdditiveIdentity
      = -1 * ::std::numeric_limits<Element>::infinity();

  using MultiplicationOp = multiplies<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(1);

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    maximum<Array<Element, 1>>, Element, 1>;
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
  arch::OpClassSimt,
  binary_or<Element>,
  binary_and<Element>,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using AdditionOp = binary_or<Element>;
  static Element constexpr AdditiveIdentity
      = static_cast<Element>(false);

  using MultiplicationOp = binary_and<Element>;
  static Element constexpr MultiplicativeIdentity
      = static_cast<Element>(true);

  using EpilogueOutputOp = epilogue::thread::SemiringAdditionOp<
    binary_or<Array<Element, 1>>, Element, 1>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

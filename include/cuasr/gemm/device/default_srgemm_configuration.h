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
  typename RingOp,
  typename OperatorClass,
  typename ArchTag
>
struct DefaultSemiRingConfiguration;

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// SM 50 //////////////////////////////////////
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
  plus_multiplies<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = cuasr::plus_multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
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
  min_plus<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = cuasr::min_plus<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
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
  max_plus<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = max_plus<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
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
  max_min<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = max_min<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
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
  min_max<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = min_max<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
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
  min_multiplies<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = min_multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
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
  max_multiplies<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = max_multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
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
  or_and<Element>,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<16, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = or_and<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// SM 80 //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Plus-Times semi-ring GEMM configuration
// this is the traditional GEMM
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  plus_multiplies<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = plus_multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

// Min-Plus (tropical) semi-ring GEMM configuration
// example application: All Pairs Shorted Path
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  min_plus<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = min_plus<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

// Max-Plus semi-ring GEMM configuration
// example application: Viterbi algorithm
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  max_plus<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = max_plus<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

// Max-Min
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  max_min<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = max_min<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

// Min-Max
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  min_max<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = min_max<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

// Min-Times
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  min_multiplies<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = min_multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

// Max-Times
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  max_multiplies<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = max_multiplies<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

// Or-And boolean ring
template <typename Element>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  or_and<Element>,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = or_and<Element>;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

} // namespace device
} // namespace gemm
} // namespace cuasr

////////////////////////////////////////////////////////////////////////////////

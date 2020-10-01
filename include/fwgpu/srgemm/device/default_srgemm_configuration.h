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

#include "fwgpu/srgemm/arch/srmma.h"
#include "cutlass/gemm/gemm.h"

#include "fwgpu/srgemm/epilogue/thread/min_op.h"

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

// Floyd–Warshall
// Semi-ring default configuration for min-plus (tropical) semi-ring
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

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<64, 128, 8>;
  using WarpShape = GemmShape<16, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using AdditionOp = minimum<Element>;
  using MultiplicationOp = plus<Element>;
  using EpilogueOutputOp = epilogue::thread::MinOp<Element, 1>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

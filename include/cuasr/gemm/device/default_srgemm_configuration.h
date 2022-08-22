/***************************************************************************************************
 * Copyright (c) 2022, Vijay Thakkar (thakkarv@gatech.edu).
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

template <typename Element, typename RingOp_, typename ArchTag>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  RingOp_,
  cutlass::arch::OpClassSimt,
  ArchTag> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 2;

  using RingOp = RingOp_;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// SM 80 //////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename RingOp_>
struct DefaultSemiRingConfiguration<
  Element,
  Element,
  Element,
  Element,
  RingOp_,
  cutlass::arch::OpClassSimt,
  cutlass::arch::Sm80> {

  static int constexpr kAlignmentA = 1;
  static int constexpr kAlignmentB = 1;
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  static int constexpr kStages = 3;

  using RingOp = RingOp_;

  using EpilogueOutputOp = cuasr::epilogue::thread::SemiringLinearCombination<
    RingOp, Element, 1>;
};

} // namespace device
} // namespace gemm
} // namespace cuasr

////////////////////////////////////////////////////////////////////////////////

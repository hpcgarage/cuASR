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
 *
 **************************************************************************************************/
/*! \file
    \brief
      Default kernel-level SRGEMM definitions combine threadblock-scoped matrix srmma
      with the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "cuasr/arch/srmma.h"
#include "cuasr/gemm/kernel/srgemm.h"
#include "cuasr/gemm/threadblock/default_srmma.h"

////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Ring operation that performs FMA
    typename RingOp_,
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    /// Use zfill or predicate for out-of-bound cp.async
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear
        = cutlass::gemm::SharedMemoryClearOption::kNone>
struct DefaultSrgemm;

////////////////////////////////////////////////////////////////////////////////

// SM50 SIMT Two Stage
template <
    /// Ring operation that performs FMA
    typename RingOp,
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial
>
struct DefaultSrgemm<
    RingOp,
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    ThreadblockShape,
    WarpShape,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    2,
    SplitKSerial> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Srmma = typename cuasr::gemm::threadblock::DefaultSrmma<
      RingOp,
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm50,
      ThreadblockShape,
      WarpShape,
      cutlass::gemm::GemmShape<1, 1, 1>,
      2>::ThreadblockSrmma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Srmma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using SrgemmKernel = cuasr::gemm::kernel::Srgemm<
      Srmma,
      RingOp,
      Epilogue,
      ThreadblockSwizzle,
      SplitKSerial
  >;
};

////////////////////////////////////////////////////////////////////////////////

// SM80 SIMT Multi Stage
template <
    /// Ring operation that performs FMA
    typename RingOp,
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial
>
struct DefaultSrgemm<
    RingOp,
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    SplitKSerial> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Srmma = typename cuasr::gemm::threadblock::DefaultSrmma<
      RingOp,
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80,
      ThreadblockShape,
      WarpShape,
      cutlass::gemm::GemmShape<1, 1, 1>,
      Stages>::ThreadblockSrmma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Srmma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using SrgemmKernel = cuasr::gemm::kernel::Srgemm<
      Srmma,
      RingOp,
      Epilogue,
      ThreadblockSwizzle,
      SplitKSerial
  >;
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cuasr

////////////////////////////////////////////////////////////////////////////////

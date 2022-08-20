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
    \brief Templates exposing architecture support for multiply-add operations
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace arch {

/// Matrix product operator for all semi-rings
template <
  /// Size of the matrix product (concept: GemmShape)
  typename Shape_,
  /// Number of threads participating
  int kThreads_,
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
  typename LayoutC,
  /// Ring operator that performa FMA
  typename RingOp
>
struct Srmma;


/////////////////////////////////////////////////////////////////////////////////////////////////

/// Semi-rings multiply-add specialized for 1 element per instruction
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
    typename LayoutC,
    /// Ring operator that performa FMA
    typename RingOp>
struct Srmma<
    cutlass::gemm::GemmShape<1, 1, 1>,
    1,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    RingOp> {
  using Shape = cutlass::gemm::GemmShape<1, 1, 1>;

  // semi-ring operators must be default contructible and
  // have a binary invocation () operator
  RingOp ring_op;

  CUTLASS_HOST_DEVICE
  void operator()(
    cutlass::Array<ElementC, 1> &d,
    cutlass::Array<ElementA, 1> const &a,
    cutlass::Array<ElementB, 1> const &b,
    cutlass::Array<ElementC, 1> const &c
  ) {
    ring_op.fma(d[0], a[0], b[0], c[0]);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////

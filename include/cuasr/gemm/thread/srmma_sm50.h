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
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
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
  /// Ring operation that performs FMA
  typename RingOp_
>
struct SrmmaGeneric {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
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

  /// Underlying semi-ring operator
  using RingOp = RingOp_;

  /// A operand storage
  using FragmentA = cutlass::Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = cutlass::Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = cutlass::Array<ElementC, Shape::kMN>;

  /// Instruction
  using SrmmaOp = RingOp;

  static bool const kMultipleOf2 = ((Shape::kM % 2 == 0) && (Shape::kN % 2 == 0));

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
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
      reinterpret_cast<ElementC *>(&D), LayoutC::packed(cutlass::make_Coord(Shape::kM, Shape::kN)));

    SrmmaOp srmma_op;

    // Copy accumulators
    D = C;

    // Compute matrix product
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < Shape::kK; ++k) {
      #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 860)
      if (kMultipleOf2 && cutlass::platform::is_same<ElementA, float>::value
                       && cutlass::platform::is_same<ElementB, float>::value
                       && cutlass::platform::is_same<ElementC, float>::value) {

        //2x2 zigzag - m and n loops to increment by 2. Inner loop to process 4 multiply-adds in a 2x2 tile.
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; n+=2) {

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < Shape::kM; m+=2) {

            int m_serpentine = (n % 4) ? (Shape::kM - 2 - m) : m;

            //top-left element in 2x2 tile
            {
              cutlass::MatrixCoord mn(m_serpentine, n);
              cutlass::MatrixCoord mk(m_serpentine, k);
              cutlass::MatrixCoord kn(k, n);
              srmma_op.fma(d_ref.at(mn), a_ref.at(mk), b_ref.at(kn), d_ref.at(mn));
            }

            //bottom-left element in 2x2 tile
            {
              cutlass::MatrixCoord mn(m_serpentine+1, n);
              cutlass::MatrixCoord mk(m_serpentine+1, k);
              cutlass::MatrixCoord kn(k, n);
              srmma_op.fma(d_ref.at(mn), a_ref.at(mk), b_ref.at(kn), d_ref.at(mn));
            }

            //bottom-right element in 2x2 tile
            {
              cutlass::MatrixCoord mn(m_serpentine+1, n+1);
              cutlass::MatrixCoord mk(m_serpentine+1, k);
              cutlass::MatrixCoord kn(k, n+1);
              srmma_op.fma(d_ref.at(mn), a_ref.at(mk), b_ref.at(kn), d_ref.at(mn));
            }

            //top-right element in 2x2 tile
            {
              cutlass::MatrixCoord mn(m_serpentine, n+1);
              cutlass::MatrixCoord mk(m_serpentine, k);
              cutlass::MatrixCoord kn(k, n+1);
              srmma_op.fma(d_ref.at(mn), a_ref.at(mk), b_ref.at(kn), d_ref.at(mn));
            }
          }
        }
      } else
      #endif
      {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < Shape::kN; ++n) {

          CUTLASS_PRAGMA_UNROLL
          for (int m = 0; m < Shape::kM; ++m) {

            int m_serpentine = (n % 2) ? (Shape::kM - 1 - m) : m;

            cutlass::MatrixCoord mn(m_serpentine, n);
            cutlass::MatrixCoord mk(m_serpentine, k);
            cutlass::MatrixCoord kn(k, n);
            srmma_op.fma(d_ref.at(mn), a_ref.at(mk), b_ref.at(kn), d_ref.at(mn));
          }
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Gemplate that handles conventional layouts for FFMA and DFMA GEMM
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
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
  /// Ring operation that performs FMA
  typename RingOp_
>
struct Srmma {

  /// Size of the Gemm problem - concept: gemm::GemmShape<>
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

  /// Ring operation that performs FMA
  using RingOp = RingOp_;

  /// A operand storage
  using FragmentA = Array<ElementA, Shape::kMK>;

  /// B operand storage
  using FragmentB = Array<ElementB, Shape::kKN>;

  /// C operand storage
  using FragmentC = Array<ElementC, Shape::kMN>;

  //
  // Methods
  //

  /// Computes a matrix product D = A * B + C
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
      RingOp> srmma;

    srmma(D, A, B, C);
  }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace gemm
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////

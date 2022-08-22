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
  \brief Kernel performing a reduction over densely packed tensors in global memory
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

namespace cuasr {
namespace reduction {
namespace thread {

/// Mixed-precision reduction with a functional reduction operator
template <
  typename RingOp_,
  typename ElementAccumulator_,
  typename Element_,
  int Count = 1
>
struct SemiringReduce {
  // Type aliases
  using RingOp = RingOp_;
  using ElementAccumulator = ElementAccumulator_;
  using Element = Element_;

  // Static members
  static int const kCount = Count;
  static Element constexpr Identity = RingOp::AddIdentity;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentElement = cutlass::Array<Element, kCount>;

  // Types nested
  struct Params { };

  // Data members
  Params params;

  /// Constructor
  CUTLASS_HOST_DEVICE
  SemiringReduce(Params params) : params(params) { };

  /// Operator
  CUTLASS_HOST_DEVICE
  FragmentAccumulator operator()(
    FragmentAccumulator accumulator,
    FragmentElement element) const {

    RingOp ring_op;
    cutlass::NumericArrayConverter<
      ElementAccumulator,
      Element,
      kCount,
      cutlass::PreferredRoundingMode<ElementAccumulator, Element>::kRound> converter;

    FragmentAccumulator retval;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      retval[i] = ring_op.add(accumulator[i], converter(element)[i]);
    }
    return retval;
  }
};

} // namespace thread
} // namespace reduction
} // namespace cuasr

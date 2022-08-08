/***************************************************************************************************
 * Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Defines basic thread level reduction with specializations for Array<T, N>.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/half.h"
#include "cutlass/functional.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace reduction {
namespace thread {

// Structure to compute the thread level reduction with semiring addition operator
template <typename RingOp, typename T, int N = 1>
struct Reduce {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    RingOp ring_op;
    return ring_op.add(lhs, rhs);
  }

  CUTLASS_HOST_DEVICE
  cutlass::Array<T, 1> operator()(cutlass::Array<T, N> const &in) const {
    cutlass::Array<T, 1> result;
    result.fill(RingOp::AddIdentity);

    CUTLASS_PRAGMA_UNROLL
    for (auto i = 0; i < N; ++i) {
      result[0] = this->operator()(result[0], in[i]);
    }

    return result;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace reduction
} // namespace cuasr

/////////////////////////////////////////////////////////////////////////////////////////////////

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
    \brief Defines basic semi-ring reels together with their identity and
    annihilator constants given type T.
*/

#pragma once

#include "cuasr/arch/functional.h"

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include <limits>
#include <type_traits>

namespace cuasr {
using cutlass::Array;

namespace {

// helpers to get the +inf/-inf and min/max for integrals/floats
// NOTE: we only use min/max values even for floats for now to avoid
// having to use actual +inf/-inf-ies.  In practice, min/max for
//  floats should behave the same as +inf/-inf
template <typename T>
constexpr auto get_inf() noexcept {
  return std::numeric_limits<T>::max();
}

template <typename T>
constexpr auto get_neginf() noexcept {
  return std::numeric_limits<T>::min();
}

} // namespace

///////////////////////////////////////////////////////////////////////////////

// Regular FMA
template <typename T>
struct plus_mult {
  static T constexpr AddIdentity     = static_cast<T>(0);
  static T constexpr MultIdentity    = static_cast<T>(1);
  static T constexpr MultAnnihilator = static_cast<T>(0);

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return lhs + rhs;
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return lhs * rhs;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct min_plus {
  static T constexpr AddIdentity     = get_inf<T>();
  static T constexpr MultIdentity    = static_cast<T>(0);
  static T constexpr MultAnnihilator = get_inf<T>();

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return cuasr::arch::min(lhs, rhs);
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return lhs * rhs;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct max_plus {
  static T constexpr AddIdentity     = get_neginf<T>();
  static T constexpr MultIdentity    = static_cast<T>(0);
  static T constexpr MultAnnihilator = get_inf<T>();

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return cuasr::arch::max(lhs, rhs);
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return lhs + rhs;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct min_mult {
  static T constexpr AddIdentity     = get_inf<T>();
  static T constexpr MultIdentity    = static_cast<T>(1);
  static T constexpr MultAnnihilator = static_cast<T>(0);

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return cuasr::arch::min(lhs, rhs);
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return lhs * rhs;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct max_mult {
  static T constexpr AddIdentity     = get_neginf<T>();
  static T constexpr MultIdentity    = static_cast<T>(1);
  static T constexpr MultAnnihilator = static_cast<T>(0);

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return cuasr::arch::max(lhs, rhs);
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return lhs * rhs;
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct min_max {
  static T constexpr AddIdentity     = get_inf<T>();
  static T constexpr MultIdentity    = get_neginf<T>();
  static T constexpr MultAnnihilator = get_inf<T>();

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return cuasr::arch::min(lhs, rhs);
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return cuasr::arch::max(lhs, rhs);
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct max_min {
  static T constexpr AddIdentity     = get_neginf<T>();
  static T constexpr MultIdentity    = get_inf<T>();
  static T constexpr MultAnnihilator = get_neginf<T>();

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return cuasr::arch::max(lhs, rhs);
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return cuasr::arch::min(lhs, rhs);
  }
};

///////////////////////////////////////////////////////////////////////////////

template <typename T>
struct or_and {
  static T constexpr AddIdentity     = static_cast<T>(0);
  static T constexpr MultIdentity    = static_cast<T>(1);
  static T constexpr MultAnnihilator = static_cast<T>(0);

  __host__ __device__
  void fma(T& dst, T const lhs, T const rhs, T const src) const {
    dst = add(src, mult(lhs, rhs));
  }

  __host__ __device__
  T add(T const lhs, T const rhs) const {
    return lhs || rhs;
  }

  __host__ __device__
  T mult(T const lhs, T const rhs) const {
    return lhs && rhs;
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cuasr

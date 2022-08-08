/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
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

/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Defines basic semi-ring reels together with their identity and
    annihilator constants given type T.

    This is inspired by the Standard Library's <functional> header.
*/

#pragma once

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
}

template <typename T, int N = 1>
struct plus {
  static T constexpr Identity    = static_cast<T>(0);
  static T constexpr Annihilator = get_inf<T>();

  // scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(scalar, rhs[i]);
    }
    return result;
  }
};

template <typename T, int N = 1>
struct multiplies {
  static T constexpr Identity    = static_cast<T>(1);
  static T constexpr Annihilator = static_cast<T>(0);

  // scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(scalar, rhs[i]);
    }
    return result;
  }
};

template <typename T, int N = 1>
struct minimum {
  static T constexpr Identity    = get_inf<T>();
  static T constexpr Annihilator = get_neginf<T>();

  // scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const { return (rhs < lhs ? rhs : lhs); }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(scalar, rhs[i]);
    }
    return result;
  }
};

template <typename T, int N = 1>
struct maximum {
  static T constexpr Identity    = get_neginf<T>();
  static T constexpr Annihilator = get_inf<T>();

  // scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const { return (lhs < rhs ? rhs : lhs); }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(scalar, rhs[i]);
    }
    return result;
  }
};

template <typename T, int N = 1>
struct binary_and {
  static T constexpr Identity    = static_cast<T>(true);
  static T constexpr Annihilator = static_cast<T>(false);

  // scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const { return lhs && rhs; }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(scalar, rhs[i]);
    }
    return result;
  }
};

template <typename T, int N = 1>
struct binary_or {
  static T constexpr Identity    = static_cast<T>(false);
  static T constexpr Annihilator = static_cast<T>(true);

  // scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const { return lhs || rhs; }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = this->operator()(scalar, rhs[i]);
    }
    return result;
  }
};

} // namespace cuasr

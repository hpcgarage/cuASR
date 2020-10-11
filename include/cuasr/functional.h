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

// PLUS operator
template <typename T>
struct plus {
  static T constexpr Identity    = static_cast<T>(0);
  static T constexpr Annihilator = get_inf<T>();

  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs += rhs;
    return lhs;
  }
};

template <typename T>
struct multiplies {
  static T constexpr Identity    = static_cast<T>(1);
  static T constexpr Annihilator = static_cast<T>(0);

  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    lhs *= rhs;
    return lhs;
  }
};

template <typename T>
struct minimum {
  static T constexpr Identity    = get_inf<T>();
  static T constexpr Annihilator = get_neginf<T>();

  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const { return (rhs < lhs ? rhs : lhs); }
};

template <>
struct minimum<float> {
  static float constexpr Identity    = get_inf<float>();
  static float constexpr Annihilator = get_neginf<float>();

  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const { return fminf(lhs, rhs); }
};

template <typename T>
struct maximum {
  static T constexpr Identity    = get_neginf<T>();
  static T constexpr Annihilator = get_inf<T>();

  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const { return (lhs < rhs ? rhs : lhs); }
};

template <>
struct maximum<float> {
  static float constexpr Identity    = get_neginf<float>();
  static float constexpr Annihilator = get_inf<float>();

  CUTLASS_HOST_DEVICE
  float operator()(float const &lhs, float const &rhs) const { return fmaxf(lhs, rhs); }
};

// binary and operator
template <typename T>
struct binary_and {
  static T constexpr Identity    = static_cast<T>(true);
  static T constexpr Annihilator = static_cast<T>(false);

  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const { return lhs && rhs; }
};

// binary or operator
template <typename T>
struct binary_or {
  static T constexpr Identity    = static_cast<T>(false);
  static T constexpr Annihilator = static_cast<T>(true);

  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const { return lhs || rhs; }
};

//
// Operators For Packed Arrays
//

template <typename T, int N>
struct plus<Array<T, N>> {
  static T constexpr Identity    = plus<T>::Identity;
  static T constexpr Annihilator = plus<T>::Annihilator;

  // expose base scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    plus<T> scalar_op;
    return scalar_op(lhs, rhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    plus<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct multiplies<Array<T, N>> {
  static T constexpr Identity    = multiplies<T>::Identity;
  static T constexpr Annihilator = multiplies<T>::Annihilator;

  // expose base scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    multiplies<T> scalar_op;
    return scalar_op(lhs, rhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    multiplies<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct minimum<Array<T, N>> {
  static T constexpr Identity    = minimum<T>::Identity;
  static T constexpr Annihilator = minimum<T>::Annihilator;

  // expose base scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    minimum<T> scalar_op;
    return scalar_op(lhs, rhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    minimum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    minimum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    minimum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct maximum<Array<T, N>> {
  static T constexpr Identity    = maximum<T>::Identity;
  static T constexpr Annihilator = maximum<T>::Annihilator;

  // expose base scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    maximum<T> scalar_op;
    return scalar_op(lhs, rhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    maximum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    maximum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }

    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    maximum<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }

    return result;
  }
};

template <typename T, int N>
struct binary_and<Array<T, N>> {
  static T constexpr Identity    = binary_and<T>::Identity;
  static T constexpr Annihilator = binary_and<T>::Annihilator;

  // expose base scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    binary_and<T> scalar_op;
    return scalar_op(lhs, rhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    binary_and<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    binary_and<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    binary_and<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }
    return result;
  }
};

template <typename T, int N>
struct binary_or<Array<T, N>> {
  static T constexpr Identity    = binary_or<T>::Identity;
  static T constexpr Annihilator = binary_or<T>::Annihilator;

  // expose base scalar operator
  CUTLASS_HOST_DEVICE
  T operator()(T const &lhs, T const &rhs) const {
    binary_or<T> scalar_op;
    return scalar_op(lhs, rhs);
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, Array<T, N> const &rhs) const {
    Array<T, N> result;
    binary_or<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], rhs[i]);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &lhs, T const &scalar) const {
    Array<T, N> result;
    binary_or<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(lhs[i], scalar);
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(T const &scalar, Array<T, N> const &rhs) const {
    Array<T, N> result;
    binary_or<T> scalar_op;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = scalar_op(scalar, rhs[i]);
    }
    return result;
  }
};

} // namespace cuasr

/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Define basic numeric operators with specializations for Array<T, N>. SIMD-ize where possible.

    This is inspired by the Standard Library's <functional> header.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include <type_traits>

namespace cutlass {

// binary and operator
template <typename T>
struct binary_and {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    return lhs && rhs;
  }
};

template <typename T, int N>
struct binary_and<Array<T, N>> {
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

// binary or operator
template <typename T>
struct binary_or {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs, T const &rhs) const {
    return lhs || rhs;
  }
};

template <typename T, int N>
struct binary_or<Array<T, N>> {
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

} // namespace cutlass

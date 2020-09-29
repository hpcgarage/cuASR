/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a min operator to an array of elements.
template <
  typename Element_,           ///< Data type used to load and store tensors
  int Count                    ///< Number of elements computed per operation
>
class MinOp {
public:
  using Element = Element_;
  using ElementOutput = Element_;

  static int const kCount = Count;

  using Fragment = Array<Element, kCount>;
  using Operator = minimum<Fragment>;

  /// Host-constructable parameters structure
  struct Params {
    bool do_min_with_source;

    CUTLASS_HOST_DEVICE
    Params(): do_min_with_source(false) { }

    CUTLASS_HOST_DEVICE
    Params(bool do_min_with_source): do_min_with_source(do_min_with_source) { }
  };

private:

  /// min operator
  Operator min_operator;
  bool do_min_with_source_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  MinOp(Params const &params) {
    do_min_with_source_ = params.do_min_with_source;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return do_min_with_source_;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition) { }

  /// Computes the min with source if configured, otherwise stores accum
  CUTLASS_HOST_DEVICE
  Fragment operator()(Fragment const &accum, Fragment const &source) const {
    return min_operator(accum, source);
  }

  /// Min not needed with source, just pass value through
  CUTLASS_HOST_DEVICE
  Fragment operator()(Fragment const &accum) const {
    return accum;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

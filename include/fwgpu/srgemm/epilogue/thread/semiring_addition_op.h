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

/// Applies an arbitrary addition operator to an array of elements.
template <
  typename AdditionOp_,        ///< Addition reel of this semi-ring
  typename Element_,           ///< Data type used to load and store tensors
  int Count                    ///< Number of elements computed per operation
>
class SemiringAdditionOp {
public:
  using Element = Element_;
  using ElementOutput = Element_;

  static int const kCount = Count;

  using Fragment = Array<Element, kCount>;
  using Operator = AdditionOp_;

  /// Host-constructable parameters structure
  struct Params {
    bool do_add_with_source;

    CUTLASS_HOST_DEVICE
    Params(): do_add_with_source(false) { }

    CUTLASS_HOST_DEVICE
    Params(bool do_add_with_source): do_add_with_source(do_add_with_source) { }
  };

private:

  /// min operator
  Operator addition_operator;
  bool do_add_with_source_;

public:

  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  SemiringAdditionOp(Params const &params) {
    do_add_with_source_ = params.do_add_with_source;
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return do_add_with_source_;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition) { }

  /// Computes the addition with source
  CUTLASS_HOST_DEVICE
  Fragment operator()(Fragment const &accum, Fragment const &source) const {
    return addition_operator(accum, source);
  }

  /// Add not needed with source, just pass value through
  CUTLASS_HOST_DEVICE
  Fragment operator()(Fragment const &accum) const {
    return accum;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cutlass

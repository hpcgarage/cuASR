/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).
 **************************************************************************************************/
/*! \file
  \brief Functor performing linear combination operations used by epilogues.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies a linear combination operator to an array of elements.
///
/// D = alpha * accumulator + beta * source + uniform
///
template <
    typename RingOp_,           ///< Ring operator that exposes .add and .mult methods
    typename ElementOutput_,    ///< Data type used to load and store tensors
    int Count,                  ///< Number of elements computed per operation
    typename ElementAccumulator_ = ElementOutput_, ///< Accumulator data type
    typename ElementCompute_
    = ElementOutput_,           ///< Data type used to compute linear combination
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class SemiringLinearCombination {
public:
  using RingOp = RingOp_;

  using ElementOutput      = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute     = ElementCompute_;
  static int const kCount  = Count;

  using FragmentOutput      = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using ComputeFragment     = cutlass::Array<ElementCompute, kCount>;

  static cutlass::FloatRoundStyle const kRound = Round;

  /// Host-constructable parameters structure
  struct Params {
    ElementCompute alpha; ///< scales accumulators
    ElementCompute beta;  ///< scales source tensor
    ElementCompute const
        *alpha_ptr; ///< pointer to accumulator scalar - if not null, loads it from memory
    ElementCompute const
        *beta_ptr; ///< pointer to source scalar - if not null, loads it from memory

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : alpha(RingOp::MultIdentity)
        , beta(RingOp::MultAnnihilator)
        , alpha_ptr(nullptr)
        , beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta)
        : alpha(alpha)
        , beta(beta)
        , alpha_ptr(nullptr)
        , beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha)
        : alpha(alpha)
        , beta(RingOp::MultAnnihilator)
        , alpha_ptr(nullptr)
        , beta_ptr(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr, ElementCompute const *beta_ptr)
        : alpha(RingOp::MultIdentity)
        , beta(RingOp::MultAnnihilator)
        , alpha_ptr(alpha_ptr)
        , beta_ptr(beta_ptr) { }

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const *alpha_ptr)
        : alpha(RingOp::MultIdentity)
        , beta(RingOp::MultAnnihilator)
        , alpha_ptr(alpha_ptr)
        , beta_ptr(nullptr) { }
  };

private:
  // scalars
  ElementCompute alpha_;
  ElementCompute beta_;
  RingOp ring_op_;

public:
  /// Constructs the function object, possibly loading from pointers in host memory
  CUTLASS_HOST_DEVICE
  SemiringLinearCombination(Params const &params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_  = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    ElementCompute kAdditiveIdentity = RingOp::AddIdentity;
    ElementCompute kMultiplicativeIdentity = RingOp::MultIdentity;

    // no source needed if mult_op(beta, C[i,j]) is equal to add_op's identity
    return (kAdditiveIdentity != ring_op_.mult(beta_, kMultiplicativeIdentity));
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition) {
    if (k_partition) {
      ElementCompute kMultiplicativeIdentity = RingOp::MultIdentity;
      beta_ = kMultiplicativeIdentity;
    }
  }

  /// Computes semiring linear scale and translate
  /// D = ring_op_.add(ring_op_.mult(alpha * accumulator), ring_op_.mult(beta * source))
  CUTLASS_HOST_DEVICE
  FragmentOutput
  operator()(FragmentAccumulator const &accumulator, FragmentOutput const &source) const {
    // Convert source to internal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round>
        source_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    ComputeFragment converted_source      = source_converter(source);
    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    // X = beta * C
    ComputeFragment intermediate;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      intermediate[i] = ring_op_.mult(beta_, converted_source[i]);
    }

    // D = (alpha * Accum) + X
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      intermediate[i] = ring_op_.add(ring_op_.mult(alpha_, converted_accumulator[i]), intermediate[i]);
    }

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }

  /// Computes semiring linear scaling: D = ring_op_.mult(alpha, accumulator)
  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const &accumulator) const {
    // Convert source to internal compute numeric type
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round>
        accumulator_converter;

    ComputeFragment converted_accumulator = accumulator_converter(accumulator);

    // Perform binary operations
    ComputeFragment intermediate;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kCount; ++i) {
      intermediate[i] = ring_op_.mult(alpha_, converted_accumulator[i]); // D = alpha * Accum
    }

    // Convert to destination numeric type
    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round>
        destination_converter;

    return destination_converter(intermediate);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace thread
} // namespace epilogue
} // namespace cuasr

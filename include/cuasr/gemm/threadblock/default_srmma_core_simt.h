/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Defines basic properties needed by CTA-level GEMMs assuming expectations about data
      layout of the global memory fragments, data types, and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting simt instructions.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"

#include "cuasr/gemm/threadblock/default_srmma_core.h"
#include "cuasr/gemm/warp/srmma_simt.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace threadblock {

namespace detail {

// convert a WarpShape which is the whole tile of elements into warp num threads.
// The goal is for each thread's tile of elements to be as square as possible
// for performance (4x4 will be faster than 2x8).
template<typename WarpShape>
constexpr int simt_get_warp_threads_m() {
    return (WarpShape::kM > WarpShape::kN) ? 8 : 4;
}

/// Computes padding in shared memory to perform efficient transpose without bank conflicts.
constexpr int simt_transpose_padding(int threads, int crosswise, int size_in_bits) {
  return (size_in_bits >= 32 ?
      threads / crosswise / (size_in_bits / 32) :
      threads / crosswise * (32 / size_in_bits)
  );
}

}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_,
                      cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::RowMajor,
                      ElementC_, LayoutC_, cutlass::arch::OpClassSimt,
                      AdditionOp_, MultiplicationOp_, 2
                       > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp  /// Multiplication operator of the semi-ring
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: column-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_,
                      cutlass::layout::RowMajor, ElementB_, cutlass::layout::ColumnMajor,
                      ElementC_, LayoutC_, cutlass::arch::OpClassSimt,
                      AdditionOp_, MultiplicationOp_, 2
                     > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    SmemThreadMapA // was IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    SmemThreadMapB // was IteratorThreadMapA
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementB>::value);

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp  /// Multiplication operator of the semi-ring
  >;


  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<kPaddingN, 0>,    // skew for A matrix to avoid SMEM bank conflicts
    cutlass::MatrixShape<0, kPaddingN>,    // skew for B matrix to avoid SMEM bank conflicts
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: row-major
///   B: row-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_,
                      cutlass::layout::RowMajor, ElementB_, cutlass::layout::RowMajor, ElementC_,
                      LayoutC_, cutlass::arch::OpClassSimt, AdditionOp_, MultiplicationOp_, 2
                       > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    SmemThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementA>::value);

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp  /// Multiplication operator of the semi-ring
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<kPaddingM, 0>,    // skew for A matrix to avoid SMEM bank conflicts
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: column-major
///   Operator: simt class
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 1>, ElementA_,
                      cutlass::layout::ColumnMajor, ElementB_, cutlass::layout::ColumnMajor,
                      ElementC_, LayoutC_, cutlass::arch::OpClassSimt,
                      AdditionOp_, MultiplicationOp_, 2
                       > {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// ThreadMap of iterator B
  using IteratorThreadMapB =  cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    SmemThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementB>::value);

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;
  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp  /// Multiplication operator of the semi-ring
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, kPaddingN>, // skew for B matrix to avoid SMEM bank conflicts
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Partial specialization:
///
///   A: column-major
///   B: row-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, int8_t,
                      cutlass::layout::ColumnMajor, int8_t, cutlass::layout::RowMajor, ElementC_,
                      LayoutC_, cutlass::arch::OpClassSimt, AdditionOp_, MultiplicationOp_, 2
                       > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = cutlass::layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;


  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp, /// Multiplication operator of the semi-ring
      PartitionsK       /// Number of partitions along K dimension
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization:
//
///
///   A: Row-major
///   B: Column-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, int8_t,
                      cutlass::layout::RowMajor, int8_t, cutlass::layout::ColumnMajor, ElementC_,
                      LayoutC_, cutlass::arch::OpClassSimt, AdditionOp_, MultiplicationOp_, 2
                       > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;


  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = cutlass::layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    SmemThreadMapA
  >;


  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = cutlass::transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    SmemThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp, /// Multiplication operator of the semi-ring
      PartitionsK       /// Number of partitions along K dimension
  >;

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementB>::value);

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<kPaddingM, 0>,
    cutlass::MatrixShape<0, kPaddingN>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization:
//
///
///   A: Row-major
///   B: Row-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, int8_t,
                      cutlass::layout::RowMajor, int8_t, cutlass::layout::RowMajor, ElementC_,
                      LayoutC_, cutlass::arch::OpClassSimt, AdditionOp_, MultiplicationOp_, 2
                       > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;


  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = cutlass::layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    SmemThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp, /// Multiplication operator of the semi-ring
      PartitionsK       /// Number of partitions along K dimension
  >;

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementB>::value);

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<kPaddingM, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Partial specialization:
//
///
///   A: Column-major
///   B: Column-major
///   Operator: simt class, for dp4a
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Addition operator of the semi-ring
    typename AdditionOp_,
    /// Multiplication operator of the semi-ring
    typename MultiplicationOp_>
struct DefaultSrmmaCore<Shape_, WarpShape_, cutlass::gemm::GemmShape<1, 1, 4>, int8_t,
                      cutlass::layout::ColumnMajor, int8_t, cutlass::layout::ColumnMajor, ElementC_,
                      LayoutC_, cutlass::arch::OpClassSimt, AdditionOp_, MultiplicationOp_, 2
                       > {

  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = int8_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = cutlass::arch::OpClassSimt;
  static int const PartitionsK = Shape::kK / WarpShape::kK;

  /// Underlying semi-ring operators
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    PartitionsK
  >;

  // Divisility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajorInterleaved<4>;
  using SmemLayoutB = cutlass::layout::RowMajorInterleaved<4>;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;


  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinear2DThreadTileStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    cutlass::layout::PitchLinearShape<4, 4>
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapB = cutlass::transform::TransposePitchLinearThreadMap2DThreadTile<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileIterator2dThreadTile<
    cutlass::MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    SmemThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = detail::simt_get_warp_threads_m<WarpShape>();
  static const int WarpNumThreadsN = kWarpSize / WarpNumThreadsM;
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(4, ThreadTileM);
  static const int LaneN = cutlass::const_min(4, ThreadTileN);
  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      4>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::ColumnMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;


  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
      WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
      ElementA,         /// Data type of A elements
      SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
      ElementB,         /// Data type of B elements
      SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
      ElementC,         /// Element type of C matrix
      LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
      Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
      AdditionOp,       /// Addition operator of the semi-ring
      MultiplicationOp, /// Multiplication operator of the semi-ring
      PartitionsK       /// Number of partitions along K dimension
  >;

  static int const kPaddingM = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementA>::value);
  static int const kPaddingN = detail::simt_transpose_padding(kWarpSize, Shape::kK, cutlass::sizeof_bits<ElementB>::value);

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, kPaddingN>,
    WarpCount::kK
  >;
};

} // namespace threadblock
} // namespace gemm
} // namespace cuasr
